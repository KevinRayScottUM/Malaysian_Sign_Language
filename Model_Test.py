import os
import json
from pathlib import Path
from collections import deque
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

# =========================
# EDIT THESE
# =========================
DATA_DIR = "/Users/kevin/PycharmProjects/TensorCat/CV/Sign_Language_Model/NPY Dataset"
CKPT_PATH = "/Users/kevin/PycharmProjects/TensorCat/CV/Sign_Language_Model/best_bilstm_attn.pt"
VIDEO_PATH = "/Users/kevin/PycharmProjects/TensorCat/CV/Sign_Language_Model/BIM_Split_6_2_2/test/baik_2/baik_2_01_03_02.mp4"

SEQ_LEN = 30
INPUT_SIZE = 258
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.35

# ---- Display options ----
WINDOW_NAME = "Sign Language"
DRAW_LANDMARKS = True          # True: draw MediaPipe pose+hands
SHOW_TOPK = 1                  # 1: only top-1, >1: show top-k
PRED_EVERY_N_FRAMES = 2        # predict every N frames (speedup)
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2

# =========================
# Device
# =========================
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

device = get_device()
print("Using device:", device)

# =========================
# Load label_map.json
# =========================
def load_gestures(label_map_path: str):
    label_map = json.loads(Path(label_map_path).read_text(encoding="utf-8"))
    items = [(int(v), k) for k, v in label_map.items()]  # (idx, name)
    items.sort(key=lambda x: x[0])
    return [k for _, k in items]

LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_map.json")
gestures = load_gestures(LABEL_MAP_PATH)
num_classes = len(gestures)
print("Loaded gestures:", num_classes)
print("First 10 gestures:", gestures[:10])

# Try infer GT label from path (.../val/<class>/<file>.mp4)
def infer_gt_from_path(video_path: str) -> str:
    p = Path(video_path)
    if len(p.parents) >= 2:
        return p.parent.name
    return "UNKNOWN"

GT_LABEL = infer_gt_from_path(VIDEO_PATH)

# =========================
# Load norm stats (CRITICAL)
# =========================
norm_path = os.path.join(DATA_DIR, "norm_stats.npz")
if not os.path.exists(norm_path):
    raise FileNotFoundError(f"Missing {norm_path}. You MUST have it from training script.")
norm = np.load(norm_path)
feat_mean = norm["mean"].astype(np.float32)  # (1,1,258)
feat_std  = norm["std"].astype(np.float32)   # (1,1,258)
print("✅ Loaded norm stats:", norm_path)

def normalize_seq_np(x_seq: np.ndarray) -> np.ndarray:
    return (x_seq - feat_mean) / feat_std

# =========================
# Model
# =========================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):
        a = torch.tanh(self.proj(h))
        score = self.v(a).squeeze(-1)
        w = torch.softmax(score, dim=1)
        out = torch.sum(h * w.unsqueeze(-1), dim=1)
        return out, w

class BetterLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.attn = AttentionPooling(hidden_size * 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        h = self.norm(h)
        ctx, _ = self.attn(h)
        return self.mlp(ctx)

model = BetterLSTM(INPUT_SIZE, HIDDEN, num_classes, num_layers=LAYERS, dropout=DROPOUT).to(device)
state_dict = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.eval()
print("✅ Loaded weights OK (strict=True):", os.path.basename(CKPT_PATH))

# =========================
# Optional: NPY sanity check
# =========================
X_test = np.load(os.path.join(DATA_DIR, "X_TEST_2.npy")).astype(np.float32)
y_test = np.load(os.path.join(DATA_DIR, "y_TEST_2.npy")).astype(np.int64)
X_test = normalize_seq_np(X_test)

with torch.no_grad():
    xb = torch.tensor(X_test, dtype=torch.float32, device=device)
    logits = model(xb)
    pred = logits.argmax(dim=1).cpu().numpy()
acc = (pred == y_test).mean()
print(f"✅ NPY sanity Test Accuracy = {acc:.4f}")

# =========================
# MediaPipe -> keypoints 258
# =========================
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

def mediapipe_detection(image_bgr, holistic):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)
    image_rgb.flags.writeable = True
    return results

def extract_keypoints_258(results) -> np.ndarray:
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark],
                        dtype=np.float32).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

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

    return np.concatenate([pose, lh, rh], axis=0)

# =========================
# UI helpers
# =========================
def put_text_with_bg(img, text, org, scale=0.7, thickness=2,
                     text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.5):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    y0 = y - h - baseline - 4
    x0 = x
    x1 = x + w + 8
    y1 = y + 4
    y0 = max(0, y0)
    x1 = min(img.shape[1] - 1, x1)
    y1 = min(img.shape[0] - 1, y1)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.putText(img, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, text_color, thickness, cv2.LINE_AA)

def format_topk(probs: torch.Tensor, k: int):
    k = min(k, probs.numel())
    vals, idxs = torch.topk(probs, k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        out.append((gestures[int(i)], float(v)))
    return out

# =========================
# Video demo
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps_src = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

print("\nStart video demo:", VIDEO_PATH)
print("GT label (from path):", GT_LABEL)
print("Press: [q]=quit, [space]=pause/resume, [s]=screenshot")

sequence = deque(maxlen=SEQ_LEN)
frame_id = 0
paused = False

# prediction cache
last_pred = "..."
last_prob = 0.0
last_topk = []

# FPS calc
t_prev = time.time()
fps_smooth = 0.0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_id += 1

            results = mediapipe_detection(frame, holistic)

            if DRAW_LANDMARKS:
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                    )
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            kp = extract_keypoints_258(results)
            sequence.append(kp)

            # predict (only when buffer full + every N frames)
            if len(sequence) == SEQ_LEN and (frame_id % PRED_EVERY_N_FRAMES == 0):
                x = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)  # (1,30,258)
                x = normalize_seq_np(x)
                xb = torch.tensor(x, dtype=torch.float32, device=device)

                with torch.no_grad():
                    logits = model(xb)[0]
                    probs = torch.softmax(logits, dim=0)

                last_topk = format_topk(probs, SHOW_TOPK)
                last_pred, last_prob = last_topk[0]

                # ✅ ONLY change: print final TOP-1 pred to console
                print(f"[frame={frame_id}] Pred: {last_pred} | prob={last_prob:.4f}")

        # FPS update (even when paused, keep stable)
        t_now = time.time()
        dt = max(1e-6, t_now - t_prev)
        inst_fps = 1.0 / dt
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth > 0 else inst_fps
        t_prev = t_now

        # overlay text
        y = 30
        put_text_with_bg(frame, f"GT: {GT_LABEL}", (10, y),
                         scale=TEXT_SCALE, thickness=TEXT_THICKNESS,
                         text_color=(255, 255, 255), bg_color=(20, 80, 20), alpha=0.55)
        y += 28
        put_text_with_bg(frame, f"Pred: {last_pred}  |  prob={last_prob:.3f}", (10, y),
                         scale=TEXT_SCALE, thickness=TEXT_THICKNESS,
                         text_color=(255, 255, 255), bg_color=(80, 20, 20), alpha=0.55)
        y += 28

        if SHOW_TOPK > 1 and len(last_topk) > 1:
            for i, (name, p) in enumerate(last_topk[1:], start=2):
                put_text_with_bg(frame, f"Top-{i}: {name} ({p:.3f})", (10, y),
                                 scale=0.6, thickness=2,
                                 text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.45)
                y += 22

        # status line
        buf = len(sequence)
        if total_frames > 0:
            status = f"frame: {frame_id}/{total_frames} | buffer: {buf}/{SEQ_LEN} | FPS: {fps_smooth:.1f}"
        else:
            status = f"frame: {frame_id} | buffer: {buf}/{SEQ_LEN} | FPS: {fps_smooth:.1f}"
        put_text_with_bg(frame, status, (10, frame.shape[0] - 12),
                         scale=0.6, thickness=2,
                         text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.45)

        if paused:
            put_text_with_bg(frame, "PAUSED (space to resume)", (10, 90),
                             scale=0.8, thickness=2,
                             text_color=(0, 0, 0), bg_color=(255, 255, 255), alpha=0.65)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            out_dir = Path("demo_outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"frame_{frame_id:05d}_{last_pred}.jpg"
            cv2.imwrite(str(out_path), frame)
            print("📸 Saved:", out_path)

cap.release()
cv2.destroyAllWindows()
print("Done.")