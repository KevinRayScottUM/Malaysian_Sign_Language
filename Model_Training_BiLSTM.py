import os
import time
import random
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix


# =========================================================
# 0) Seed
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)


# =========================================================
# 1) Device
# =========================================================
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
# 2) Load NPY
# =========================================================
DATA_DIR = "NPY Dataset"

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


# =========================================================
# 2.0) label_map -> target_names
# =========================================================
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_map.json")
if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)  # {"class_name": idx}
    idx2name = {int(v): k for k, v in label_map.items()}
    target_names = [idx2name.get(i, str(i)) for i in range(num_classes)]
else:
    label_map = None
    target_names = [str(i) for i in range(num_classes)]


# =========================================================
# 2.1) Normalize (save stats!)
# =========================================================
feat_mean = X_train.mean(axis=(0, 1), keepdims=True)        # (1,1,D)
feat_std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-6  # (1,1,D)

# 保存到文件：Model_Test / 视频推理必须加载并使用它
np.savez(os.path.join(DATA_DIR, "norm_stats.npz"), mean=feat_mean, std=feat_std)
print("✅ Saved norm stats ->", os.path.join(DATA_DIR, "norm_stats.npz"))

X_train = (X_train - feat_mean) / feat_std
X_val   = (X_val   - feat_mean) / feat_std
X_test  = (X_test  - feat_mean) / feat_std


# =========================================================
# 2.2) class weights
# =========================================================
counts = np.bincount(y_train, minlength=num_classes)
print("Train class count: min =", counts.min(), "max =", counts.max(),
      "imbalance ratio =", counts.max() / max(1, counts.min()))

class_weights = (counts.sum() / (counts + 1e-6))
class_weights = np.sqrt(class_weights)
class_weights = class_weights / class_weights.mean()
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)


# =========================================================
# 2.3) DataLoader
# =========================================================
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(device.type == "cuda"))
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))


# =========================================================
# 3) Model
# =========================================================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):  # (B,T,H)
        a = torch.tanh(self.proj(h))
        score = self.v(a).squeeze(-1)      # (B,T)
        w = torch.softmax(score, dim=1)    # (B,T)
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

hidden_size = 128
model = BetterLSTM(input_size, hidden_size, num_classes, num_layers=2, dropout=0.35).to(device)
print(model)


# =========================================================
# 4) Loss / Optim / Scheduler
# =========================================================
train_criterion  = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.1)
report_criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
GRAD_CLIP = 1.0


# =========================================================
# 5) Eval
# =========================================================
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


# =========================================================
# 6) Train loop + save best AFTER EACH EPOCH (end of epoch)
#     best file is PURE state_dict to avoid torch.load security issues
# =========================================================
num_epochs = 200
patience = 20
best_val_loss = float("inf")
best_epoch = -1
no_improve = 0

BEST_WEIGHTS_PATH = "best_bilstm_attn.pt"

train_losses, val_losses, test_losses = [], [], []
train_accuracies, val_accuracies, test_accuracies = [], [], []
val_macro_f1s, test_macro_f1s = [], []

global_start = time.time()

for epoch in range(1, num_epochs + 1):
    epoch_start = time.time()
    model.train()

    running_rep_loss = 0.0
    correct, total = 0, 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

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

    train_losses.append(train_loss); train_accuracies.append(train_acc)
    val_losses.append(val_loss);     val_accuracies.append(val_acc)
    test_losses.append(test_loss);   test_accuracies.append(test_acc)
    val_macro_f1s.append(val_f1);    test_macro_f1s.append(test_f1)

    scheduler.step(val_loss)

    # ✅ 保存发生在“一个epoch完整结束之后”
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_epoch = epoch
        no_improve = 0

        # 只保存权重：Model_Test 直接 load_state_dict 即可
        torch.save(model.state_dict(), BEST_WEIGHTS_PATH)

        # 顺便存一份元信息（不参与 torch.load）
        meta = {
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "hidden_size": hidden_size,
            "num_layers": 2,
            "dropout": 0.35,
            "input_size": int(input_size),
            "seq_len": int(seq_len),
            "num_classes": int(num_classes),
        }
        with open("best_bilstm_attn_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    else:
        no_improve += 1

    epoch_time = time.time() - epoch_start
    elapsed = time.time() - global_start
    avg_epoch = elapsed / epoch
    eta = avg_epoch * (num_epochs - epoch)

    lr_now = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch [{epoch:3d}/{num_epochs}] "
        f"train_loss:{train_loss:.4f} val_loss:{val_loss:.4f} test_loss:{test_loss:.4f} | "
        f"train_acc:{train_acc:.4f} val_acc:{val_acc:.4f} test_acc:{test_acc:.4f} | "
        f"val_F1:{val_f1:.4f} test_F1:{test_f1:.4f} | "
        f"lr:{lr_now:.2e} time:{format_time(epoch_time)} ETA:{format_time(eta)} "
        f"{'(best)' if epoch == best_epoch else ''}"
    )

    if no_improve >= patience:
        print(f"\nEarly stopping at epoch={epoch}. Best epoch={best_epoch}, best val_loss={best_val_loss:.4f}")
        break


# =========================================================
# 7) Final eval with best weights
# =========================================================
print("\nLoading best weights:", BEST_WEIGHTS_PATH)
model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location=device))
final_test_loss, final_test_acc, yt, pt = run_eval(test_loader, return_preds=True)

print(f"\nBest Epoch: {best_epoch} | Final Test Loss: {final_test_loss:.4f}, Acc: {final_test_acc:.4f}")
print("Final Test F1 (macro):   ", f1_score(yt, pt, average="macro", zero_division=0))
print("Final Test F1 (weighted):", f1_score(yt, pt, average="weighted", zero_division=0))

print("\n=== Per-class Report (Test) ===")
print(classification_report(yt, pt, target_names=target_names, digits=2, zero_division=0))

cm = confusion_matrix(yt, pt, labels=list(range(num_classes)))
print("Confusion Matrix shape:", cm.shape)


# =========================================================
# 8) Curves
# =========================================================
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label="Train loss")
plt.plot(epochs, val_losses,   label="Val loss")
plt.plot(epochs, test_losses,  label="Test loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Train/Val/Test Loss")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracies, label="Train acc")
plt.plot(epochs, val_accuracies,   label="Val acc")
plt.plot(epochs, test_accuracies,  label="Test acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Train/Val/Test Accuracy")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, val_macro_f1s,  label="Val macro-F1")
plt.plot(epochs, test_macro_f1s, label="Test macro-F1")
plt.xlabel("Epoch"); plt.ylabel("Macro-F1")
plt.title("Val/Test Macro-F1")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# =========================================================
# 9) Full Confusion Matrix table print (NO shape-only)
# =========================================================
import numpy as np

# 1) Ensure numpy prints the FULL matrix (no "...")
np.set_printoptions(
    threshold=np.inf,   # never summarize
    linewidth=2000,     # avoid line wrapping (increase if needed)
    suppress=True
)

print("\n=== Confusion Matrix (FULL, Test) ===")
print(cm)  # full numeric matrix

# 2) Pretty table with labels (truncate labels to keep it readable)
def _short(s, n=12):
    s = str(s)
    return s if len(s) <= n else s[:n-1] + "…"

labels = [_short(n, 12) for n in target_names]  # keep original target_names unchanged

# Header
cell_w = max(4, max(len(x) for x in labels))  # width for label cells
num_w = max(3, len(str(int(cm.max()))) if cm.size else 3)  # width for numbers

header = " " * (cell_w + 2) + " ".join(f"{lab:>{num_w}}" for lab in labels)
print("\n" + header)

# Rows
for i, lab in enumerate(labels):
    row_nums = " ".join(f"{int(v):>{num_w}d}" for v in cm[i])
    print(f"{lab:<{cell_w}} | {row_nums}")

print("\n(Rows=True, Cols=Pred). If the table wraps, increase linewidth or reduce label length.)")