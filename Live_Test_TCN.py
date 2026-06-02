import os
import json
import time

import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr


# =========================================================
# CONFIG (edit here)
# =========================================================
DATA_DIR = "NPY Dataset"

WEIGHTS_PATH = "best_tcn_causal.pt"
META_PATH = "best_tcn_causal_meta.json"
NORM_PATH = os.path.join(DATA_DIR, "norm_stats.npz")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_map.json")

SEQ_LEN = 30
FEATURE_DIM = 258

# smoothing
MIN_PROB_TO_ACCEPT = 0.60
CONFIRM_FRAMES = 6
EMA_ALPHA = 0.88

# performance
FRAME_RESIZE_WIDTH = 960
PROCESS_EVERY_N_FRAMES = 1

# mediapipe
MP_DET_CONF = 0.5
MP_TRK_CONF = 0.5


# =========================================================
# Device
# =========================================================
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

device = get_device()
print("Using device:", device)


# =========================================================
# Causal TCN (must match training)
# =========================================================
class CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            c_in, c_out, kernel_size=kernel_size,
            dilation=dilation, padding=0, bias=bias
        )

    def forward(self, x):
        # left-pad only => causal
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
    def __init__(self, input_dim, num_classes, channels=256, levels=6, kernel_size=3, dropout=0.25):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, channels)
        self.blocks = nn.ModuleList()
        for i in range(levels):
            dilation = 2 ** i
            self.blocks.append(
                TemporalBlock(channels, channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            )
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
        h = h.transpose(1, 2).contiguous()  # (B,C,T)
        for blk in self.blocks:
            h = blk(h)
        last = h[:, :, -1]                  # causal last step
        return self.head(last)


# =========================================================
# Load meta + norm + label map + weights
# =========================================================
for p in [META_PATH, WEIGHTS_PATH, NORM_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

num_classes = int(meta["num_classes"])
input_size = int(meta["input_size"])
channels = int(meta.get("channels", 256))
levels = int(meta.get("levels", 6))
kernel_size = int(meta.get("kernel_size", 3))
dropout = float(meta.get("dropout", 0.25))

norm = np.load(NORM_PATH)
feat_mean = norm["mean"].astype(np.float32).reshape(1, 1, -1)
feat_std  = norm["std"].astype(np.float32).reshape(1, 1, -1)

idx2name = {}
if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx2name = {int(v): k for k, v in label_map.items()}

def id_to_name(i: int) -> str:
    return idx2name.get(int(i), str(i))

model = CausalTCNClassifier(
    input_dim=input_size,
    num_classes=num_classes,
    channels=channels,
    levels=levels,
    kernel_size=kernel_size,
    dropout=dropout
).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()
print("Loaded:", WEIGHTS_PATH)


# =========================================================
# MediaPipe (GLOBAL singleton; do NOT put in gr.State)
# =========================================================
mp_holistic = mp.solutions.holistic
HOLISTIC = mp_holistic.Holistic(
    min_detection_confidence=MP_DET_CONF,
    min_tracking_confidence=MP_TRK_CONF
)

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

    feat = np.concatenate([pose, lh, rh], axis=0)
    if feat.shape[0] != FEATURE_DIM:
        out = np.zeros((FEATURE_DIM,), dtype=np.float32)
        out[:min(FEATURE_DIM, feat.shape[0])] = feat[:min(FEATURE_DIM, feat.shape[0])]
        return out
    return feat


# =========================================================
# State (deepcopy-able only)
# =========================================================
def make_state():
    return {
        "buf": [],
        "ema_prob": None,
        "last_raw_id": None,
        "same_count": 0,
        "stable_id": None,
        "frame_idx": 0,
        "fps": 0.0,
        "t_last": time.time(),
    }

def reset_state():
    return make_state(), "Reset done."


def smoother_update(st, prob: np.ndarray):
    if st["ema_prob"] is None:
        ema = prob
    else:
        ema_prev = np.array(st["ema_prob"], dtype=np.float32)
        ema = EMA_ALPHA * ema_prev + (1.0 - EMA_ALPHA) * prob

    st["ema_prob"] = ema.tolist()

    raw_id = int(np.argmax(ema))
    raw_p = float(np.max(ema))

    if raw_p < MIN_PROB_TO_ACCEPT:
        st["last_raw_id"] = None
        st["same_count"] = 0
        return None, raw_p, raw_id

    if st["last_raw_id"] == raw_id:
        st["same_count"] += 1
    else:
        st["last_raw_id"] = raw_id
        st["same_count"] = 1

    if st["same_count"] >= CONFIRM_FRAMES:
        st["stable_id"] = raw_id

    return st["stable_id"], raw_p, raw_id


# =========================================================
# Streaming inference
# =========================================================
@torch.no_grad()
def stream_infer(frame_rgb: np.ndarray, st: dict):
    if st is None:
        st = make_state()
    if frame_rgb is None:
        return None, "No frame.", st

    st["frame_idx"] += 1
    frame_idx = st["frame_idx"]

    now = time.time()
    dt = now - st["t_last"]
    if dt > 0:
        st["fps"] = 0.9 * st["fps"] + 0.1 * (1.0 / dt)
    st["t_last"] = now

    if FRAME_RESIZE_WIDTH is not None and frame_rgb.shape[1] > FRAME_RESIZE_WIDTH:
        scale = FRAME_RESIZE_WIDTH / frame_rgb.shape[1]
        frame_rgb = cv2.resize(frame_rgb, (FRAME_RESIZE_WIDTH, int(frame_rgb.shape[0] * scale)))

    do_process = (PROCESS_EVERY_N_FRAMES <= 1) or (frame_idx % PROCESS_EVERY_N_FRAMES == 0)

    stable_name = "warming up"
    stable_prob = 0.0
    raw_name = "-"
    raw_prob = 0.0

    if do_process:
        results = HOLISTIC.process(frame_rgb)
        feat = extract_keypoints_258(results)

        st["buf"].append(feat.tolist())
        if len(st["buf"]) > SEQ_LEN:
            st["buf"] = st["buf"][-SEQ_LEN:]

        if len(st["buf"]) == SEQ_LEN:
            seq = np.array(st["buf"], dtype=np.float32).reshape(1, SEQ_LEN, -1)
            seq = (seq - feat_mean) / feat_std

            xb = torch.from_numpy(seq).to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)

            stable_id, p_now, raw_id = smoother_update(st, prob)
            raw_name = id_to_name(raw_id)
            raw_prob = float(p_now)

            if stable_id is None:
                stable_name = "unknown"
                stable_prob = float(p_now)
            else:
                stable_name = id_to_name(stable_id)
                stable_prob = float(p_now)

    overlay_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = overlay_bgr.shape[:2]
    cv2.rectangle(overlay_bgr, (10, 10), (w - 10, 130), (0, 0, 0), -1)
    cv2.putText(overlay_bgr, f"Stable: {stable_name} | prob={stable_prob:.3f}",
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(overlay_bgr, f"Window: {len(st['buf'])}/{SEQ_LEN} | Raw: {raw_name} ({raw_prob:.3f})",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(overlay_bgr, f"FPS: {st['fps']:.1f} | device: {device.type}",
                (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    panel = (
        f"Frame: {frame_idx}\n"
        f"Buffer: {len(st['buf'])}/{SEQ_LEN}\n"
        f"Stable: {stable_name}\n"
        f"Stable prob: {stable_prob:.4f}\n"
        f"Raw: {raw_name}\n"
        f"Raw prob (EMA max): {raw_prob:.4f}\n"
        f"FPS: {st['fps']:.2f}\n"
        f"Device: {device.type}\n"
        f"Model: Causal TCN\n"
    )

    return overlay_rgb, panel, st


# =========================================================
# Gradio UI
# =========================================================
with gr.Blocks(title="Live Gesture - Causal TCN") as demo:
    gr.Markdown("## Real-time Gesture Recognition (Webcam) - Causal TCN (1D Temporal CNN)")

    state = gr.State(make_state())

    with gr.Row():
        cam = gr.Image(
            sources=["webcam"],
            streaming=True,
            type="numpy",
            label="Webcam (streaming)"
        )
        out_text = gr.Textbox(label="Prediction Panel", lines=12)

    with gr.Row():
        btn_reset = gr.Button("Reset")
        status = gr.Textbox(label="Status", value="Ready.", interactive=False)

    cam.stream(
        fn=stream_infer,
        inputs=[cam, state],
        outputs=[cam, out_text, state],
        show_progress=False
    )

    btn_reset.click(
        fn=reset_state,
        inputs=None,
        outputs=[state, status],
        show_progress=False
    )

if __name__ == "__main__":
    # Gradio version compatibility
    try:
        demo.launch(queue=False)
    except TypeError:
        demo.launch()