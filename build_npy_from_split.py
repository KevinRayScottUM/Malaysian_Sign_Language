import os
import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# =========================
# EDIT THESE
# =========================
SPLIT_ROOT = "/Users/kevin/PycharmProjects/TensorCat/CV/Sign_Language_Model/BIM_Split_6_2_2"
OUT_DIR    = "/Users/kevin/PycharmProjects/TensorCat/CV/Sign_Language_Model/NPY Dataset"

SEQ_LEN = 30
FEATURE_DIM = 258

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

    return np.concatenate([pose, lh, rh], axis=0)  # (258,)

def sample_frame_indices(n_frames: int, seq_len: int) -> np.ndarray:
    if n_frames <= 0:
        return np.array([], dtype=np.int32)
    if n_frames == 1:
        return np.zeros((seq_len,), dtype=np.int32)
    idx = np.linspace(0, n_frames - 1, seq_len)
    return np.round(idx).astype(np.int32)

def video_to_sequence(video_path: str, holistic) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # 读不了视频 -> 返回全0
        return np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = sample_frame_indices(n_frames, SEQ_LEN)

    seq = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)

    for t, fid in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        results = mediapipe_detection(frame, holistic)
        seq[t] = extract_keypoints_258(results)

    cap.release()
    return seq

def list_classes(split_root: str) -> list[str]:
    train_dir = Path(split_root) / "train"
    classes = [p.name for p in train_dir.iterdir() if p.is_dir()]
    classes.sort()
    return classes

def collect_split(split_name: str, classes: list[str], split_root: str):
    X_list, y_list, path_list = [], [], []

    split_dir = Path(split_root) / split_name
    for ci, cname in enumerate(classes):
        class_dir = split_dir / cname
        if not class_dir.exists():
            continue
        vids = sorted([p for p in class_dir.rglob("*.mp4")])
        for vp in vids:
            X_list.append(str(vp))
            y_list.append(ci)

    return X_list, np.array(y_list, dtype=np.int64)

def build():
    os.makedirs(OUT_DIR, exist_ok=True)

    classes = list_classes(SPLIT_ROOT)
    label_map = {c: i for i, c in enumerate(classes)}
    with open(os.path.join(OUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print("Classes:", len(classes))
    print("First 10:", classes[:10])

    train_paths, y_train = collect_split("train", classes, SPLIT_ROOT)
    val_paths,   y_val   = collect_split("val",   classes, SPLIT_ROOT)
    test_paths,  y_test  = collect_split("test",  classes, SPLIT_ROOT)

    # 保存路径（便于你溯源）
    Path(os.path.join(OUT_DIR, "paths_train.txt")).write_text("\n".join(train_paths), encoding="utf-8")
    Path(os.path.join(OUT_DIR, "paths_val.txt")).write_text("\n".join(val_paths), encoding="utf-8")
    Path(os.path.join(OUT_DIR, "paths_test.txt")).write_text("\n".join(test_paths), encoding="utf-8")

    print("train/val/test counts:", len(train_paths), len(val_paths), len(test_paths))

    mp_h = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def build_X(paths, tag):
        X = np.zeros((len(paths), SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        for i, p in enumerate(paths):
            if (i + 1) % 50 == 0:
                print(f"[{tag}] {i+1}/{len(paths)}")
            X[i] = video_to_sequence(p, mp_h)
        return X

    X_train = build_X(train_paths, "train")
    X_val   = build_X(val_paths,   "val")
    X_test  = build_X(test_paths,  "test")

    np.save(os.path.join(OUT_DIR, "X_TRAIN_2.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "y_TRAIN_2.npy"), y_train)

    np.save(os.path.join(OUT_DIR, "X_VAL_2.npy"), X_val)
    np.save(os.path.join(OUT_DIR, "y_VAL_2.npy"), y_val)

    np.save(os.path.join(OUT_DIR, "X_TEST_2.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "y_TEST_2.npy"), y_test)

    mp_h.close()
    print("✅ Done. Saved to:", OUT_DIR)

if __name__ == "__main__":
    build()