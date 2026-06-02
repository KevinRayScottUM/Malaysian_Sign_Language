import os
import time
import json
import random
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import f1_score, classification_report, confusion_matrix


# =========================================================
# 0) CONFIG (edit here, no argparse)
# =========================================================
DATA_DIR = "NPY Dataset"  # where X_*.npy, y_*.npy, label_map.json are
SEQ_LEN = 30
FEATURE_DIM = 258

# ---- training ----
DO_TRAIN = True                 # True: train uni-LSTM and save best weights
DO_LIVE  = True                 # True: run webcam real-time demo after training (or directly if weights exist)

BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 20
LR = 3e-3
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.35

BEST_WEIGHTS_PATH = "best_uni_lstm_attn.pt"
BEST_META_PATH = "best_uni_lstm_attn_meta.json"

# ---- real-time ----
CAM_INDEX = 0
MIN_PROB_TO_ACCEPT = 0.60       # if max prob below this => "unknown"
CONFIRM_FRAMES = 6              # need same prediction for N consecutive frames to switch label
EMA_ALPHA = 0.85                # higher => smoother, more stable (0.8~0.95 common)
SHOW_FPS = True
PRINT_CONSOLE_LOG = True

# MediaPipe
MP_DET_CONF = 0.5
MP_TRK_CONF = 0.5

# performance tips
PROCESS_EVERY_N_FRAMES = 1      # set 2 or 3 if too slow
FRAME_RESIZE_WIDTH = 960        # resize input for speed; set None to disable


# =========================================================
# 1) Utils
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

device = get_device()
print("Using device:", device)

use_amp = (device.type == "cuda")
if use_amp and hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=True)
else:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)

def format_time(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


# =========================================================
# 2) MediaPipe feature extraction (258 dims)
# =========================================================
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image_bgr, holistic):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)
    image_rgb.flags.writeable = True
    return results

def extract_keypoints_258(results) -> np.ndarray:
    # pose: 33 * 4 = 132
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark],
                        dtype=np.float32).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    # hands: 21 * 3 = 63 each
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark],
                      dtype=np.float32).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark],
                      dtype=np.float32).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    feat = np.concatenate([pose, lh, rh], axis=0)  # (258,)
    if feat.shape[0] != FEATURE_DIM:
        # safety
        out = np.zeros((FEATURE_DIM,), dtype=np.float32)
        out[:min(FEATURE_DIM, feat.shape[0])] = feat[:min(FEATURE_DIM, feat.shape[0])]
        return out
    return feat


# =========================================================
# 3) Load label_map (for readable names)
# =========================================================
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_map.json")
if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)  # {"class_name": idx}
    idx2name = {int(v): k for k, v in label_map.items()}
else:
    label_map = None
    idx2name = {}

def id_to_name(i: int) -> str:
    return idx2name.get(int(i), str(i))


# =========================================================
# 4) Model (Uni-LSTM + LayerNorm + AttentionPooling)
# =========================================================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):  # (B,T,H)
        a = torch.tanh(self.proj(h))
        score = self.v(a).squeeze(-1)   # (B,T)
        w = torch.softmax(score, dim=1) # (B,T)
        out = torch.sum(h * w.unsqueeze(-1), dim=1)  # (B,H)
        return out, w

class UniLSTM_Attn(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False  # <-- key change for real-time
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = AttentionPooling(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        h, _ = self.lstm(x)     # (B,T,H)
        h = self.norm(h)
        ctx, _ = self.attn(h)   # (B,H)
        return self.mlp(ctx)    # (B,C)


# =========================================================
# 5) Training (build uni-LSTM weights + save norm stats)
# =========================================================
def train_and_save():
    # ---- load NPY ----
    X_train = np.load(os.path.join(DATA_DIR, "X_TRAIN_2.npy")).astype(np.float32)
    y_train = np.load(os.path.join(DATA_DIR, "y_TRAIN_2.npy")).astype(np.int64)
    X_val   = np.load(os.path.join(DATA_DIR, "X_VAL_2.npy")).astype(np.float32)
    y_val   = np.load(os.path.join(DATA_DIR, "y_VAL_2.npy")).astype(np.int64)
    X_test  = np.load(os.path.join(DATA_DIR, "X_TEST_2.npy")).astype(np.float32)
    y_test  = np.load(os.path.join(DATA_DIR, "y_TEST_2.npy")).astype(np.int64)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape,   "y_val:  ", y_val.shape)
    print("X_test: ", X_test.shape,  "y_test: ", y_test.shape)

    num_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
    seq_len = X_train.shape[1]
    input_size = X_train.shape[2]
    print("seq_len =", seq_len, "input_size =", input_size, "num_classes =", num_classes)

    # ---- normalize (train stats only) ----
    feat_mean = X_train.mean(axis=(0, 1), keepdims=True)        # (1,1,D)
    feat_std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-6  # (1,1,D)
    np.savez(os.path.join(DATA_DIR, "norm_stats.npz"), mean=feat_mean, std=feat_std)
    print("Saved norm stats ->", os.path.join(DATA_DIR, "norm_stats.npz"))

    X_train = (X_train - feat_mean) / feat_std
    X_val   = (X_val   - feat_mean) / feat_std
    X_test  = (X_test  - feat_mean) / feat_std

    # ---- class weights ----
    counts = np.bincount(y_train, minlength=num_classes)
    print("Train class count: min =", counts.min(), "max =", counts.max(),
          "imbalance ratio =", counts.max() / max(1, counts.min()))

    class_weights = (counts.sum() / (counts + 1e-6))
    class_weights = np.sqrt(class_weights)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # ---- loaders ----
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # ---- model ----
    model = UniLSTM_Attn(input_size, HIDDEN_SIZE, num_classes, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    print(model)

    train_criterion  = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=LABEL_SMOOTHING)
    report_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    def run_eval(loader, return_preds=False):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = report_criterion(logits, yb)

                total_loss += loss.item() * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

                if return_preds:
                    ys.append(yb.cpu().numpy())
                    ps.append(pred.cpu().numpy())

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)

        if return_preds:
            return avg_loss, acc, np.concatenate(ys), np.concatenate(ps)
        return avg_loss, acc

    # ---- train loop ----
    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0
    global_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        running_rep_loss = 0.0
        correct, total = 0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # landmark noise augmentation (helps robustness)
            xb = xb + 0.01 * torch.randn_like(xb)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(xb)
                loss = train_criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                rep_loss = report_criterion(logits, yb)
                running_rep_loss += rep_loss.item() * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        train_loss = running_rep_loss / max(1, total)
        train_acc  = correct / max(1, total)

        val_loss,  val_acc,  yv, pv = run_eval(val_loader,  return_preds=True)
        test_loss, test_acc, yt, pt = run_eval(test_loader, return_preds=True)

        val_f1  = f1_score(yv, pv, average="macro", zero_division=0)
        test_f1 = f1_score(yt, pt, average="macro", zero_division=0)

        scheduler.step(val_loss)

        # save best by val_loss
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0

            torch.save(model.state_dict(), BEST_WEIGHTS_PATH)

            meta = {
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss),
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "input_size": int(input_size),
                "seq_len": int(seq_len),
                "num_classes": int(num_classes),
                "model": "UniLSTM_Attn",
            }
            with open(BEST_META_PATH, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        else:
            no_improve += 1

        elapsed = time.time() - global_start
        avg_epoch = elapsed / epoch
        eta = avg_epoch * (EPOCHS - epoch)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:3d}/{EPOCHS}] "
            f"train_loss:{train_loss:.4f} val_loss:{val_loss:.4f} test_loss:{test_loss:.4f} | "
            f"train_acc:{train_acc:.4f} val_acc:{val_acc:.4f} test_acc:{test_acc:.4f} | "
            f"val_F1:{val_f1:.4f} test_F1:{test_f1:.4f} | "
            f"lr:{lr_now:.2e} time:{format_time(time.time()-epoch_start)} ETA:{format_time(eta)} "
            f"{'(best)' if epoch == best_epoch else ''}"
        )

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch={epoch}. Best epoch={best_epoch}, best val_loss={best_val_loss:.4f}")
            break

    # final report on best weights
    print("\nLoading best weights:", BEST_WEIGHTS_PATH)
    model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location=device))
    model.eval()
    _, _, yt, pt = run_eval(test_loader, return_preds=True)

    # build target_names from label_map if available
    num_classes = int(np.max(yt)) + 1
    target_names = [id_to_name(i) for i in range(num_classes)]

    print("\n=== Per-class Report (Test) ===")
    print(classification_report(yt, pt, target_names=target_names, digits=2, zero_division=0))
    print("Confusion matrix shape:", confusion_matrix(yt, pt).shape)

    print("\n✅ Training done. Best weights saved:", BEST_WEIGHTS_PATH)


# =========================================================
# 6) Real-time webcam inference (sliding window + norm + smoothing)
# =========================================================
class RealTimeSmoother:
    """EMA smoothing + consecutive-frame confirmation."""
    def __init__(self, num_classes: int, ema_alpha=0.85, confirm_frames=6, min_prob=0.6):
        self.num_classes = num_classes
        self.ema_alpha = ema_alpha
        self.confirm_frames = confirm_frames
        self.min_prob = min_prob

        self.ema_prob = None
        self.last_raw_id = None
        self.same_count = 0
        self.stable_id = None

    def update(self, prob: np.ndarray):
        if self.ema_prob is None:
            self.ema_prob = prob.copy()
        else:
            self.ema_prob = self.ema_alpha * self.ema_prob + (1.0 - self.ema_alpha) * prob

        raw_id = int(np.argmax(self.ema_prob))
        raw_p = float(np.max(self.ema_prob))

        # unknown gate
        if raw_p < self.min_prob:
            self.last_raw_id = None
            self.same_count = 0
            # keep stable_id unchanged (or set to None if you prefer)
            return None, raw_p, raw_id

        # consecutive confirmation
        if self.last_raw_id == raw_id:
            self.same_count += 1
        else:
            self.last_raw_id = raw_id
            self.same_count = 1

        if self.same_count >= self.confirm_frames:
            self.stable_id = raw_id

        return self.stable_id, raw_p, raw_id


def run_live():
    # load meta to get num_classes/input_size
    if not os.path.exists(BEST_META_PATH):
        raise FileNotFoundError(f"Missing {BEST_META_PATH}. Train first or provide meta.")

    with open(BEST_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_classes = int(meta["num_classes"])
    input_size = int(meta["input_size"])

    # load norm stats
    norm_path = os.path.join(DATA_DIR, "norm_stats.npz")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing {norm_path}. Train first to generate it.")
    norm = np.load(norm_path)
    feat_mean = norm["mean"].astype(np.float32)  # (1,1,D)
    feat_std  = norm["std"].astype(np.float32)   # (1,1,D)

    # model
    model = UniLSTM_Attn(
        input_size=input_size,
        hidden_size=int(meta["hidden_size"]),
        num_classes=num_classes,
        num_layers=int(meta["num_layers"]),
        dropout=float(meta["dropout"])
    ).to(device)
    model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location=device))
    model.eval()
    print("✅ Loaded weights:", BEST_WEIGHTS_PATH)

    smoother = RealTimeSmoother(
        num_classes=num_classes,
        ema_alpha=EMA_ALPHA,
        confirm_frames=CONFIRM_FRAMES,
        min_prob=MIN_PROB_TO_ACCEPT
    )

    # sliding window of features
    feat_buf = deque(maxlen=SEQ_LEN)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check CAM_INDEX or permissions.")

    mp_h = mp_holistic.Holistic(
        min_detection_confidence=MP_DET_CONF,
        min_tracking_confidence=MP_TRK_CONF
    )

    frame_idx = 0
    last_time = time.time()
    fps = 0.0

    stable_name = "warming up"
    stable_prob = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame_idx += 1
            if FRAME_RESIZE_WIDTH is not None and frame.shape[1] > FRAME_RESIZE_WIDTH:
                scale = FRAME_RESIZE_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, (FRAME_RESIZE_WIDTH, int(frame.shape[0] * scale)))

            # optionally skip frames for speed
            if PROCESS_EVERY_N_FRAMES > 1 and (frame_idx % PROCESS_EVERY_N_FRAMES != 0):
                # still show frame and last prediction
                pass
            else:
                results = mediapipe_detection(frame, mp_h)
                feat = extract_keypoints_258(results)  # (258,)
                feat_buf.append(feat)

                if len(feat_buf) == SEQ_LEN:
                    seq = np.stack(list(feat_buf), axis=0).astype(np.float32)  # (30,258)

                    # normalize using training stats
                    seq = (seq - feat_mean.reshape(1, -1)) / feat_std.reshape(1, -1)

                    xb = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1,30,258)
                    with torch.no_grad():
                        logits = model(xb)
                        prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                    stable_id, p_stable_or_raw, raw_id = smoother.update(prob)

                    if stable_id is None:
                        stable_name = "unknown"
                        stable_prob = float(p_stable_or_raw)
                        pred_name = id_to_name(raw_id)
                        pred_prob = float(p_stable_or_raw)
                    else:
                        stable_name = id_to_name(stable_id)
                        stable_prob = float(p_stable_or_raw)
                        pred_name = stable_name
                        pred_prob = stable_prob

                    if PRINT_CONSOLE_LOG:
                        print(f"[frame={frame_idx}] Pred: {pred_name} | prob={pred_prob:.4f}")

            # fps
            now = time.time()
            dt = now - last_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last_time = now

            # overlay
            overlay = frame.copy()
            text1 = f"Stable: {stable_name}  prob={stable_prob:.3f}"
            text2 = f"Window: {len(feat_buf)}/{SEQ_LEN}  Device: {device.type}"
            cv2.rectangle(overlay, (10, 10), (frame.shape[1] - 10, 110), (0, 0, 0), -1)
            cv2.putText(overlay, text1, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(overlay, text2, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if SHOW_FPS:
                cv2.putText(overlay, f"FPS: {fps:.1f}", (frame.shape[1] - 160, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            cv2.imshow("Real-time Gesture (UniLSTM + Attn, Sliding Window)", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        mp_h.close()
        cap.release()
        cv2.destroyAllWindows()


# =========================================================
# 7) Main
# =========================================================
if __name__ == "__main__":
    if DO_TRAIN:
        train_and_save()

    if DO_LIVE:
        if not os.path.exists(BEST_WEIGHTS_PATH):
            raise FileNotFoundError(f"Missing {BEST_WEIGHTS_PATH}. Set DO_TRAIN=True first.")
        run_live()