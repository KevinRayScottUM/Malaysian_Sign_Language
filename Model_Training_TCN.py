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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import f1_score, classification_report, confusion_matrix


# =========================================================
# 0) CONFIG (edit here)
# =========================================================
DATA_DIR = "NPY Dataset"     # contains X_*.npy, y_*.npy, label_map.json
SEQ_LEN = 30                 # must match your preprocessing
FEATURE_DIM = 258

DO_TRAIN = True              # train and save best weights
DO_LIVE  = True              # webcam inference after training (or if weights exist)

# ---- training ----
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 20
LR = 3e-3
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
NOISE_STD = 0.01             # landmark noise augmentation

# ---- TCN (Temporal CNN) ----
# Causal dilated conv stack (TCN-like)
TCN_CHANNELS = 256           # hidden channels
TCN_LEVELS = 6               # dilation levels: 1,2,4,8,16,32 (enough for T=30)
KERNEL_SIZE = 3
DROPOUT = 0.25

BEST_WEIGHTS_PATH = "best_tcn_causal.pt"
BEST_META_PATH = "best_tcn_causal_meta.json"

# ---- real-time (webcam) ----
CAM_INDEX = 0
MIN_PROB_TO_ACCEPT = 0.60
CONFIRM_FRAMES = 6
EMA_ALPHA = 0.88

PROCESS_EVERY_N_FRAMES = 1   # set 2 for speed if needed
FRAME_RESIZE_WIDTH = 960     # set smaller for speed; None disables

SHOW_FPS = True
PRINT_CONSOLE_LOG = True

# MediaPipe
MP_DET_CONF = 0.5
MP_TRK_CONF = 0.5


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
        out = np.zeros((FEATURE_DIM,), dtype=np.float32)
        out[:min(FEATURE_DIM, feat.shape[0])] = feat[:min(FEATURE_DIM, feat.shape[0])]
        return out
    return feat


# =========================================================
# 3) label_map (id -> name)
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
# 4) Causal TCN Model (1D Temporal CNN)
# =========================================================
class CausalConv1d(nn.Module):
    """
    Conv1d that only pads on the left -> strictly causal.
    Input:  (B, C_in, T)
    Output: (B, C_out, T)
    """
    def __init__(self, c_in, c_out, kernel_size=3, dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size,
                              dilation=dilation, padding=0, bias=bias)

    def forward(self, x):
        # pad left only
        pad_left = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad_left, 0))
        return self.conv(x)

class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, dilation=1, dropout=0.25):
        super().__init__()
        self.conv1 = CausalConv1d(c_in, c_out, kernel_size, dilation)
        self.conv2 = CausalConv1d(c_out, c_out, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(c_out)
        self.norm2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

        self.down = None
        if c_in != c_out:
            self.down = nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.drop(out)

        res = x if self.down is None else self.down(x)
        return out + res

class CausalTCNClassifier(nn.Module):
    """
    Input x: (B, T, D)
    - project D -> C
    - causal dilated conv stack over time
    - take LAST timestep embedding (strictly causal)
    - MLP -> logits
    """
    def __init__(self, input_dim, num_classes, channels=256, levels=6, kernel_size=3, dropout=0.25):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, channels)
        self.blocks = nn.ModuleList()
        for i in range(levels):
            dilation = 2 ** i
            self.blocks.append(TemporalBlock(channels, channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout))

        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B,T,D)
        h = self.in_proj(x)                 # (B,T,C)
        h = h.transpose(1, 2).contiguous()  # (B,C,T) for conv1d

        for blk in self.blocks:
            h = blk(h)                      # (B,C,T)

        last = h[:, :, -1]                  # (B,C) ONLY uses past (causal)
        return self.head(last)              # (B,num_classes)


# =========================================================
# 5) Train + save best
# =========================================================
def train_and_save():
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

    # --- normalize using TRAIN stats only ---
    feat_mean = X_train.mean(axis=(0, 1), keepdims=True)        # (1,1,D)
    feat_std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-6  # (1,1,D)
    np.savez(os.path.join(DATA_DIR, "norm_stats.npz"), mean=feat_mean, std=feat_std)
    print("✅ Saved norm stats ->", os.path.join(DATA_DIR, "norm_stats.npz"))

    X_train = (X_train - feat_mean) / feat_std
    X_val   = (X_val   - feat_mean) / feat_std
    X_test  = (X_test  - feat_mean) / feat_std

    # --- class weights (softened with sqrt) ---
    counts = np.bincount(y_train, minlength=num_classes)
    print("Train class count: min =", counts.min(), "max =", counts.max(),
          "imbalance ratio =", counts.max() / max(1, counts.min()))

    class_weights = (counts.sum() / (counts + 1e-6))
    class_weights = np.sqrt(class_weights)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # --- loaders ---
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # --- model ---
    model = CausalTCNClassifier(
        input_dim=input_size,
        num_classes=num_classes,
        channels=TCN_CHANNELS,
        levels=TCN_LEVELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT
    ).to(device)
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

    # --- train loop ---
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

            # landmark noise augmentation
            xb = xb + NOISE_STD * torch.randn_like(xb)

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

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0

            torch.save(model.state_dict(), BEST_WEIGHTS_PATH)

            meta = {
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss),
                "seq_len": int(seq_len),
                "input_size": int(input_size),
                "num_classes": int(num_classes),
                "model": "CausalTCNClassifier",
                "channels": int(TCN_CHANNELS),
                "levels": int(TCN_LEVELS),
                "kernel_size": int(KERNEL_SIZE),
                "dropout": float(DROPOUT),
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

    # final report using best
    print("\nLoading best weights:", BEST_WEIGHTS_PATH)
    model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location=device))
    model.eval()
    _, _, yt, pt = run_eval(test_loader, return_preds=True)

    target_names = [id_to_name(i) for i in range(int(np.max(yt)) + 1)]
    print("\n=== Per-class Report (Test) ===")
    print(classification_report(yt, pt, target_names=target_names, digits=2, zero_division=0))
    print("Confusion matrix shape:", confusion_matrix(yt, pt).shape)

    print("\n✅ Training done. Best weights saved:", BEST_WEIGHTS_PATH)


# =========================================================
# 6) Real-time smoothing (EMA + confirm frames)
# =========================================================
class RealTimeSmoother:
    def __init__(self, num_classes: int, ema_alpha=0.88, confirm_frames=6, min_prob=0.60):
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

        if raw_p < self.min_prob:
            self.last_raw_id = None
            self.same_count = 0
            return None, raw_p, raw_id

        if self.last_raw_id == raw_id:
            self.same_count += 1
        else:
            self.last_raw_id = raw_id
            self.same_count = 1

        if self.same_count >= self.confirm_frames:
            self.stable_id = raw_id

        return self.stable_id, raw_p, raw_id


# =========================================================
# 7) Live webcam inference (sliding window + norm stats)
# =========================================================
def run_live():
    if not os.path.exists(BEST_META_PATH):
        raise FileNotFoundError(f"Missing {BEST_META_PATH}. Train first or provide meta.")

    with open(BEST_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_classes = int(meta["num_classes"])
    input_size = int(meta["input_size"])

    # norm stats MUST match training
    norm_path = os.path.join(DATA_DIR, "norm_stats.npz")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing {norm_path}. Train first to generate it.")
    norm = np.load(norm_path)
    feat_mean = norm["mean"].astype(np.float32).reshape(1, 1, -1)  # (1,1,D)
    feat_std  = norm["std"].astype(np.float32).reshape(1, 1, -1)   # (1,1,D)

    # model
    model = CausalTCNClassifier(
        input_dim=input_size,
        num_classes=num_classes,
        channels=int(meta["channels"]),
        levels=int(meta["levels"]),
        kernel_size=int(meta["kernel_size"]),
        dropout=float(meta["dropout"]),
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

    feat_buf = deque(maxlen=SEQ_LEN)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check CAM_INDEX or permissions.")

    mp_h = mp_holistic.Holistic(min_detection_confidence=MP_DET_CONF, min_tracking_confidence=MP_TRK_CONF)

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

            if PROCESS_EVERY_N_FRAMES > 1 and (frame_idx % PROCESS_EVERY_N_FRAMES != 0):
                pass
            else:
                results = mediapipe_detection(frame, mp_h)
                feat = extract_keypoints_258(results)
                feat_buf.append(feat)

                if len(feat_buf) == SEQ_LEN:
                    seq = np.stack(list(feat_buf), axis=0).astype(np.float32)  # (T,D)
                    seq = (seq.reshape(1, SEQ_LEN, -1) - feat_mean) / feat_std  # (1,T,D)

                    xb = torch.from_numpy(seq).to(device)  # (1,T,D)
                    with torch.no_grad():
                        logits = model(xb)
                        prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                    stable_id, p_now, raw_id = smoother.update(prob)

                    if stable_id is None:
                        stable_name = "unknown"
                        stable_prob = float(p_now)
                        pred_name = id_to_name(raw_id)
                        pred_prob = float(p_now)
                    else:
                        stable_name = id_to_name(stable_id)
                        stable_prob = float(p_now)
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

            overlay = frame.copy()
            text1 = f"Stable: {stable_name}  prob={stable_prob:.3f}"
            text2 = f"Window: {len(feat_buf)}/{SEQ_LEN}  Device: {device.type}"
            cv2.rectangle(overlay, (10, 10), (overlay.shape[1] - 10, 110), (0, 0, 0), -1)
            cv2.putText(overlay, text1, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(overlay, text2, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if SHOW_FPS:
                cv2.putText(overlay, f"FPS: {fps:.1f}", (overlay.shape[1] - 160, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            cv2.imshow("Real-time Gesture (Causal TCN, Sliding Window)", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        mp_h.close()
        cap.release()
        cv2.destroyAllWindows()


# =========================================================
# 8) Main
# =========================================================
if __name__ == "__main__":
    if DO_TRAIN:
        train_and_save()

    if DO_LIVE:
        if not os.path.exists(BEST_WEIGHTS_PATH):
            raise FileNotFoundError(f"Missing {BEST_WEIGHTS_PATH}. Set DO_TRAIN=True first.")
        run_live()