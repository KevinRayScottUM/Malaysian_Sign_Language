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
import gradio as gr

# =========================
# EDIT THESE
# =========================
DATA_DIR = "/Users/kevin/PycharmProjects/TensorCat/CV/Sign_Language_Model/NPY Dataset"
CKPT_PATH = "/Users/kevin/PycharmProjects/TensorCat/CV/Sign_Language_Model/best_bilstm_attn.pt"

SEQ_LEN = 30
INPUT_SIZE = 258
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.35

# UI/Runtime options
DRAW_LANDMARKS = True
SHOW_TOPK = 1                 # keep 1 for top-1 only
PRED_EVERY_N_FRAMES = 2       # speedup
MAX_LOG_LINES = 120           # log panel keep last N lines
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
# MediaPipe -> keypoints 258
# =========================
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

# NOTE: 作为全局单例，避免每帧反复 init（否则会很慢）
_holistic = None
def get_holistic():
    global _holistic
    if _holistic is None:
        _holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _holistic

def mediapipe_detection(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = get_holistic().process(image_rgb)
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
# UI helpers (OpenCV draw)
# =========================
def put_text_with_bg(img_bgr, text, org, scale=0.7, thickness=2,
                     text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.55):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    y0 = max(0, y - h - baseline - 6)
    x0 = max(0, x - 4)
    x1 = min(img_bgr.shape[1] - 1, x + w + 8)
    y1 = min(img_bgr.shape[0] - 1, y + 6)

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)

    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, text_color, thickness, cv2.LINE_AA)

def format_topk(probs: torch.Tensor, k: int):
    k = min(k, probs.numel())
    vals, idxs = torch.topk(probs, k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        out.append((gestures[int(i)], float(v)))
    return out

# =========================
# Gradio state
# =========================
def init_state():
    return {
        "seq": deque(maxlen=SEQ_LEN),
        "frame_id": 0,
        "logs": [],
        "last_pred": "...",
        "last_prob": 0.0,
        "last_topk": [],
        "t_prev": time.time(),
        "fps_smooth": 0.0,
    }

def clear_all():
    st = init_state()
    return None, st, "Cleared.", ""  # output_img, state, status, log_text

# =========================
# Core: process each frame
# =========================
def process_frame(frame_rgb: np.ndarray, st: dict):
    if st is None:
        st = init_state()

    if frame_rgb is None:
        return None, st, "No frame", "\n".join(st.get("logs", []))

    st["frame_id"] += 1
    fid = st["frame_id"]

    # gr.Image gives RGB -> convert to BGR for cv2
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # mediapipe
    results = mediapipe_detection(frame_bgr)

    # draw landmarks
    if DRAW_LANDMARKS:
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # keypoints -> sequence buffer
    kp = extract_keypoints_258(results)
    st["seq"].append(kp)

    # prediction (every N frames, when buffer full)
    if len(st["seq"]) == SEQ_LEN and (fid % PRED_EVERY_N_FRAMES == 0):
        x = np.expand_dims(np.array(st["seq"], dtype=np.float32), axis=0)  # (1,30,258)
        x = normalize_seq_np(x)
        xb = torch.tensor(x, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model(xb)[0]
            probs = torch.softmax(logits, dim=0)

        topk = format_topk(probs, max(1, int(SHOW_TOPK)))
        st["last_topk"] = topk
        st["last_pred"], st["last_prob"] = topk[0]

        # ✅ console log line (top-1)
        line = f"[frame={fid}] Pred: {st['last_pred']} | prob={st['last_prob']:.4f}"
        print(line)

        st["logs"].append(line)
        if len(st["logs"]) > MAX_LOG_LINES:
            st["logs"] = st["logs"][-MAX_LOG_LINES:]

    # FPS
    t_now = time.time()
    dt = max(1e-6, t_now - st["t_prev"])
    inst_fps = 1.0 / dt
    st["fps_smooth"] = 0.9 * st["fps_smooth"] + 0.1 * inst_fps if st["fps_smooth"] > 0 else inst_fps
    st["t_prev"] = t_now

    # overlay pred on TOP-RIGHT
    pred_text = f"{st['last_pred']} ({st['last_prob']:.3f})"
    (tw, th), _ = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)
    x = frame_bgr.shape[1] - tw - 20
    y = 30
    put_text_with_bg(
        frame_bgr,
        pred_text,
        (x, y),
        scale=TEXT_SCALE,
        thickness=TEXT_THICKNESS,
        text_color=(255, 255, 255),
        bg_color=(80, 20, 20),
        alpha=0.60
    )

    status = f"frame={fid} | buffer={len(st['seq'])}/{SEQ_LEN} | FPS={st['fps_smooth']:.1f}"
    log_text = "\n".join(st["logs"])

    # back to RGB for gradio
    out_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb, st, status, log_text

# =========================
# UI (Blocks)  —— 关键：log_box 在最上面 + 事件输出直连它
# =========================
CSS = """
#logbox textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 12px;
  line-height: 1.25;
}
"""

with gr.Blocks(css=CSS) as demo:
    gr.Markdown("## Live Sign Language Demo (Webcam) — BiLSTM + Attention")
    gr.Markdown(
        f"- Model: `{os.path.basename(CKPT_PATH)}`  \n"
        f"- Classes: `{num_classes}`  \n"
        f"- Sequence length: `{SEQ_LEN}`  \n"
        f"- Prediction: **Top-1 shown on video (top-right) + printed in console + appended to log panel**"
    )

    st = gr.State(init_state())

    # ✅✅✅ LOG PANEL AT THE VERY TOP (full width)
    log_box = gr.Textbox(
        label="Console-like Logs (live)",
        value="",
        interactive=False,
        lines=18,
        max_lines=300,
        elem_id="logbox"
    )

    with gr.Row():
        webcam = gr.Image(
            sources=["webcam"],
            streaming=True,
            type="numpy",
            label="Webcam"
        )
        output = gr.Image(type="numpy", label="Output (annotated)")

    with gr.Row():
        status = gr.Textbox(label="Status", interactive=False)
        clear_btn = gr.Button("Clear")

    # ✅ 关键：事件直接 outputs 到 log_box（就是上面那个 log_box）
    if hasattr(webcam, "stream"):
        webcam.stream(
            fn=process_frame,
            inputs=[webcam, st],
            outputs=[output, st, status, log_box]
        )
    else:
        # fallback（某些旧版本没有 stream，就用 change）
        webcam.change(
            fn=process_frame,
            inputs=[webcam, st],
            outputs=[output, st, status, log_box]
        )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[output, st, status, log_box]
    )

if __name__ == "__main__":
    demo.queue(False).launch()