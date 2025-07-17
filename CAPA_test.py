# test_inference.py
import os, time, pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from CNN_model import SPPNet, SimpleCNN  # 모델 정의 모듈

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if CUDA else "cpu")
BATCH  = 1024            # 원하는 배치 크기

log_path = f'./result/SimpleCNN/6x6/test_log_poly.txt'
# ────────────────────────────────────────────────────────────
# 1. 데이터셋
class FaultDataset(Dataset):
    def __init__(self, x_np, y_np):
        self.x, self.y = x_np, y_np
    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        x = self.x[idx]
        if x.ndim == 4:  # (1,H,W,1) → (H,W,1)
            x = x.squeeze(0)
        if x.shape[0] != 1:     # (H,W,1) → (1,H,W)
            x = x.transpose(2,0,1)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(self.y[idx]).float()
        return x, y

with open("./datasets_test/MLS4050/6x6/x_tr.pickle","rb") as f: x_tr = pickle.load(f)
with open("./datasets_test/MLS4050/6x6/y_tr.pickle","rb") as f: y_tr = pickle.load(f)

test_ds = FaultDataset(x_tr, y_tr)
test_loader = DataLoader(test_ds, batch_size=BATCH,
                         shuffle=False, num_workers=4, pin_memory=CUDA)

print("revised\n")
# ────────────────────────────────────────────────────────────
# 2. 모델 로드
num_rows = y_tr.shape[2]   # (N,2,H) → H,W
num_cols = y_tr.shape[2]
INNER_CH = 64; LOOP = 10              # 저장 당시 하이퍼파라미터

model = SimpleCNN(num_classes_row=num_rows, num_classes_col=num_cols, inner_channels=64, loop=10).to(DEVICE)
ckpt = torch.load("./model/SimpleCNN/6x6/SimpleCNN-epoch:264.pth", map_location=DEVICE)
model.load_state_dict(ckpt if isinstance(ckpt,dict) else ckpt.state_dict())
model.eval()

# ────────────────────────────────────────────────────────────
# 3. 정확도·추론시간 측정
def bin_accuracy(logits, target, thr=0.5):
    pred = (logits.sigmoid() > thr).float()
    return (pred.eq(target).float().mean()).item()

total_acc_row = total_acc_col = 0.0
n_batches     = 0
elapsed_ms    = 0.0

with torch.no_grad():
    for x,y in test_loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        torch.cuda.synchronize() if CUDA else None
        t0 = time.time()

        row_out, col_out = model(x)

        torch.cuda.synchronize() if CUDA else None
        elapsed_ms += (time.time() - t0) * 1000

        total_acc_row += bin_accuracy(row_out, y[:,0,:])
        total_acc_col += bin_accuracy(col_out, y[:,1,:])
        n_batches     += 1

avg_row_acc = total_acc_row / n_batches
avg_col_acc = total_acc_col / n_batches
avg_time_ms = elapsed_ms   / len(test_ds)
avg_batch_ms = elapsed_ms / n_batches

print(f"Test accuracy – Row: {avg_row_acc:.4f}  |  Col: {avg_col_acc:.4f}")
print(f"Average inference time: {avg_time_ms:.2f} ms / sample")
print(f"Average latency per batch (size={BATCH}): {avg_batch_ms:.2f} ms")

with open(log_path, 'w') as f:
    f.write(f"Test accuracy – Row: {avg_row_acc:.4f}  |  Col: {avg_col_acc:.4f}")
    f.write(f"Average inference time: {avg_time_ms:.2f} ms / sample")
    f.write(f"Average latency per batch (size={BATCH}): {avg_batch_ms:.2f} ms")
