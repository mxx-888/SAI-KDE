# ======= 解决 Windows 下常见 OpenMP 重复加载崩溃（OMP Error #15）=======
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# =====================================================================================

import math
import random
import sys
import numpy as np
import pandas as pd

import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.nn as nn

# ===================== Seed / Device =====================
def set_seed(seed=48):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(48)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===================== Columns =====================
x_cols = ['WS', 'Ta', 'RH', 'Press', 'SW_IN', 'LAI', 'SIF', 'acc_rain7', 'SM',
          'Month', 'DEM', 'GRA', 'WSA', 'SAV', 'EBF', 'DBF', 'MF', 'ENF',
          'WET', 'CRO', 'OSH', 'CSH', 'CLAY', 'SAND', 'SILT']
y_cols = ["ET", "H"]

LULC_COLS = ['GRA', 'WSA', 'SAV', 'EBF', 'DBF', 'MF', 'ENF', 'WET', 'CRO', 'OSH', 'CSH']
LULC_IDXS = [x_cols.index(c) for c in LULC_COLS]
num_groups = len(LULC_COLS)
mf_id = LULC_COLS.index("MF")
mf_feat_idx = x_cols.index("MF")  # MF one-hot 在 x 里的列号

# ===================== Seq config (CNN_LSTM必须) =====================
SEQ_LEN = 7
HORIZON = 1  # t+1

# ===================== Train config =====================
Batchsize = 128
learning_rate = 0.01
Freeze_epoch = 300

# KDE-DW
alpha_max = 0.8
alpha_mf = 0.3
eps_dw = 1e-6
w_clip_min, w_clip_max = 0.2, 2.0

# IO
path_train = r"D:\SAI\SAI\SAI-main\experiment\exp1\train"
path_val   = r"D:\SAI\SAI\SAI-main\experiment\exp1\validation"
path_maxmin = r"D:\SAI\SAI\SAI-main\experiment\exp1\maxmin"

save_dir = r"D:\SAI\SAI\SAI-main\logs-CNNLSTM-KDE-seed48"
os.makedirs(save_dir, exist_ok=True)
base_ckpt_dir = os.path.join(save_dir, "every_epoch")
os.makedirs(base_ckpt_dir, exist_ok=True)

# ===================== Utils =====================
def get_data(path):
    dfs = []
    for file in os.listdir(path):
        fp = os.path.join(path, file)
        if os.path.isfile(fp) and file.lower().endswith(".csv"):
            dfs.append(pd.read_csv(fp))
    if not dfs:
        raise FileNotFoundError(f"No csv files found in: {path}")
    return pd.concat(dfs, axis=0, ignore_index=True)

def data_norm(data):
    maxlist, minlist = [], []
    for i in range(data.shape[1]):
        col = data[:, i:i+1]
        mx = np.max(col)
        mn = np.min(col)
        maxlist.append(mx)
        minlist.append(mn)
        if mx != mn:
            data[:, i:i+1] = (col - mn) / (mx - mn)
        else:
            data[:, i:i+1] = 0.0
    return maxlist, minlist, data

def rmse_np(pred, true):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((pred - true) ** 2)))

# ============== build windows (返回 target index 用于对齐 gid/p0/真值) ==============
def build_windows_xy_with_target_idx(X, Y, seq_len=7, horizon=1):
    """
    X: (N,F)  -> Xw: (Nw,seq_len,F)
    Y: (N,D)  -> Yw: (Nw,D)  对应 target index = i+seq_len+horizon-1
    return: Xw, Yw, target_idxs
    """
    N = X.shape[0]
    end = N - seq_len - horizon + 1
    if end <= 0:
        return None, None, None
    Xw = np.stack([X[i:i+seq_len, :] for i in range(end)], axis=0)
    target_idxs = np.arange(end) + (seq_len + horizon - 1)
    Yw = Y[target_idxs, :]
    return Xw.astype(np.float32), Yw.astype(np.float32), target_idxs.astype(np.int64)

# ===================== KDE Approx (与你脚本一致) =====================
def silverman_bandwidth(y_1d: np.ndarray):
    y = np.asarray(y_1d, dtype=np.float64)
    n = len(y)
    std = np.std(y)
    h = 1.06 * std * (n ** (-1 / 5))
    return max(float(h), 1e-6)

def kde_1d_hist(y_1d, bins=512, bandwidth=None):
    y = np.asarray(y_1d, dtype=np.float64)
    if y.size == 0:
        return np.zeros((0, 1), dtype=np.float32), 1e-6

    if bandwidth is None:
        bandwidth = silverman_bandwidth(y)

    ymin, ymax = float(np.min(y)), float(np.max(y))
    if ymax <= ymin + 1e-12:
        p0 = np.ones_like(y, dtype=np.float64)
        return p0.reshape(-1, 1).astype(np.float32), float(bandwidth)

    hist, edges = np.histogram(y, bins=bins, range=(ymin, ymax), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = float(centers[1] - centers[0])

    sigma = max(bandwidth / (dx + 1e-12), 1e-6)
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= (kernel.sum() + 1e-12)

    pdf = np.convolve(hist, kernel, mode="same")
    p = np.interp(y, centers, pdf, left=pdf[0], right=pdf[-1])
    p0 = p / (np.max(p) + 1e-12)
    return p0.reshape(-1, 1).astype(np.float32), float(bandwidth)

def compute_p0_by_kde_groupwise_hist(y_raw, group_ids, num_groups, min_group_n=50, bins=512):
    p0_all = np.zeros((len(y_raw), 1), dtype=np.float32)
    p0_global, global_bw = kde_1d_hist(y_raw, bins=bins, bandwidth=None)

    for g in range(num_groups):
        idx = (group_ids == g)
        if int(idx.sum()) < min_group_n:
            p0_all[idx] = p0_global[idx]
        else:
            p0_g, _ = kde_1d_hist(y_raw[idx], bins=bins, bandwidth=None)
            p0_all[idx] = p0_g

    return p0_all, global_bw

# ===================== Model: CNN_LSTM =====================
class CNN_LSTM(nn.Module):
    """
    输入: (B,T,F)
    Conv1d: 需要先转成 (B,F,T)
    """
    def __init__(self, num_features, out_dim=2,
                 cnn_channels=64, kernel_size=3, dropout=0.2,
                 lstm_hidden=64, lstm_layers=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x):
        # x: (B,T,F)
        x = x.transpose(1, 2)  # (B,F,T)
        x = self.cnn(x)        # (B,C,T')
        x = x.transpose(1, 2)  # (B,T',C)
        out, _ = self.lstm(x)  # (B,T',H)
        last = out[:, -1, :]
        return self.fc(last)   # (B,2) normalized

# ===================== Loss: 加入 KDE 权重(只加权 ET MSE项) =====================
class MyLossDW(nn.Module):
    def __init__(self, a=0.8, c=0.1):
        super().__init__()
        self.a = a
        self.c = c

    @staticmethod
    def weighted_mean(per_sample_loss, w):
        return (per_sample_loss * w).sum() / (w.sum() + 1e-12)

    def forward(self, pred, y, w):
        # pred/y: (B,2) normalized
        # w: (B,1)
        w = w.to(pred.device)

        ET = y[:, 0:1]
        ET_pre = pred[:, 0:1]
        per_ori = (ET - ET_pre).pow(2)
        loss_ori = self.weighted_mean(per_ori, w)

        ET4 = ET.pow(4)
        ET4_pre = ET_pre.pow(4)
        loss_add = (ET4 - ET4_pre).pow(2).mean()

        H = y[:, 1:2]
        H_pre = pred[:, 1:2]
        loss_H = (H - H_pre).pow(2).mean()

        return self.a * loss_ori + (1 - self.a - self.c) * loss_add + self.c * loss_H

# ===================== Load / Normalize =====================
max_min_data = get_data(path_maxmin)
datamaxmin_x = np.array(max_min_data[x_cols]).astype(float)
maxx, minx, _ = data_norm(datamaxmin_x)

datamaxmin_y = np.array(max_min_data[y_cols]).astype(float)
maxy, miny, _ = data_norm(datamaxmin_y)

# ---- val ----
dataval = get_data(path_val)
datavalx = np.array(dataval[x_cols]).astype(float)
datavaly_raw = np.array(dataval[y_cols]).astype(float)  # raw

for i in range(datavalx.shape[1]):
    if maxx[i] != minx[i]:
        datavalx[:, i:i+1] = (datavalx[:, i:i+1] - minx[i]) / (maxx[i] - minx[i])
    else:
        datavalx[:, i:i+1] = 0.0

datavaly = datavaly_raw.copy().astype(float)  # normalized for training/val loss if needed
for i in range(datavaly.shape[1]):
    if maxy[i] != miny[i]:
        datavaly[:, i:i+1] = (datavaly[:, i:i+1] - miny[i]) / (maxy[i] - miny[i])
    else:
        datavaly[:, i:i+1] = 0.0

# ---- train ----
data = get_data(path_train)
datatrainx = np.array(data[x_cols]).astype(float)
datatrainy_raw = np.array(data[y_cols]).astype(float)  # raw
datatrainy = datatrainy_raw.copy().astype(float)       # normalized

for i in range(datatrainx.shape[1]):
    if maxx[i] != minx[i]:
        datatrainx[:, i:i+1] = (datatrainx[:, i:i+1] - minx[i]) / (maxx[i] - minx[i])
    else:
        datatrainx[:, i:i+1] = 0.0

for i in range(datatrainy.shape[1]):
    if maxy[i] != miny[i]:
        datatrainy[:, i:i+1] = (datatrainy[:, i:i+1] - miny[i]) / (maxy[i] - miny[i])
    else:
        datatrainy[:, i:i+1] = 0.0

print("train rows:", datatrainx.shape[0], "val rows:", datavalx.shape[0])

# ===================== group ids（按每一天的 LULC one-hot）=====================
train_group_ids_full = np.argmax(datatrainx[:, LULC_IDXS], axis=1).astype(np.int64)
val_group_ids_full   = np.argmax(datavalx[:,   LULC_IDXS], axis=1).astype(np.int64)

counts = np.bincount(train_group_ids_full, minlength=num_groups)
print("[Group counts]", {LULC_COLS[i]: int(counts[i]) for i in range(num_groups)})

# ===================== KDE 得到 p0（对每一天的 ET raw 做）=====================
p0_full, global_bw = compute_p0_by_kde_groupwise_hist(
    datatrainy_raw[:, 0], train_group_ids_full, num_groups, min_group_n=50, bins=512
)
print(f"[DenseWeight] Global bandwidth = {global_bw:.6g}")
print("[DenseWeight] p0 quantiles:", np.quantile(p0_full, [0.01, 0.05, 0.5, 0.95, 0.99]).ravel())

# ===================== 构造滑窗（关键：窗口样本的 gid/p0 取 target day 的）=====================
trainXw, trainYw, train_tidx = build_windows_xy_with_target_idx(datatrainx, datatrainy, SEQ_LEN, HORIZON)
valXw,   valYw,   val_tidx   = build_windows_xy_with_target_idx(datavalx,   datavaly,   SEQ_LEN, HORIZON)

if trainXw is None or valXw is None:
    raise ValueError("数据太短，无法构造序列窗口：检查 SEQ_LEN/HORIZON 或数据长度。")

train_p0_w  = p0_full[train_tidx]                          # (Nw,1)
train_gid_w = train_group_ids_full[train_tidx]             # (Nw,)
val_gid_w   = val_group_ids_full[val_tidx]                 # (Nv,)
val_mf_mask = (val_gid_w == mf_id)
val_nonmf_mask = ~val_mf_mask

# val raw 真值对齐窗口输出（用 target idx 取）
val_true_raw_cut = datavaly_raw[val_tidx, :]               # (Nv,2)

print("windowed train:", trainXw.shape, "windowed val:", valXw.shape)

# ===================== DataLoader (加入 p0, gid) =====================
train_x  = torch.from_numpy(trainXw).float()
train_y  = torch.from_numpy(trainYw).float()
train_p0 = torch.from_numpy(train_p0_w).float()
train_gid = torch.from_numpy(train_gid_w).long()

val_x = torch.from_numpy(valXw).float().to(device)
val_y_raw_cut = torch.from_numpy(val_true_raw_cut).float()  # CPU

train_ds = Data.TensorDataset(train_x, train_y, train_p0, train_gid)
train_loader = DataLoader(train_ds, batch_size=Batchsize, shuffle=True, num_workers=0)

# ===================== Train =====================
model = CNN_LSTM(num_features=len(x_cols), out_dim=len(y_cols)).to(device)
loss_fn = MyLossDW(a=0.8, c=0.1)

warm_up_epochs = int(Freeze_epoch * 0.3)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
warm_up_cosine_lr = lambda epoch: (
    epoch / warm_up_epochs if epoch <= warm_up_epochs else
    0.5 * (math.cos((epoch - warm_up_epochs) / (Freeze_epoch - warm_up_epochs) * math.pi) + 1)
)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warm_up_cosine_lr)

# 训练集窗口级的 p0/gid（用于算 mean_w）
train_p0_all = train_p0  # CPU tensor
train_gid_all = train_gid

def eval_rmse_all_mf_nonmf(model_, val_x_, val_y_raw_cut_):
    model_.eval()
    with torch.no_grad():
        val_pred_norm = model_(val_x_)  # (Nv,2) normalized
    pred_et = val_pred_norm[:, 0:1].detach().cpu().numpy()
    pred_et_raw = pred_et * (maxy[0] - miny[0]) + miny[0]
    true_et_raw = val_y_raw_cut_[:, 0:1].detach().cpu().numpy()

    rmse_all = rmse_np(pred_et_raw, true_et_raw)
    rmse_mf = rmse_np(pred_et_raw[val_mf_mask], true_et_raw[val_mf_mask]) if val_mf_mask.any() else float("nan")
    rmse_nonmf = rmse_np(pred_et_raw[val_nonmf_mask], true_et_raw[val_nonmf_mask]) if val_nonmf_mask.any() else float("nan")
    return rmse_all, rmse_mf, rmse_nonmf

patience = 50
best_val = float("inf")
counter = 0
best_model_path = None

print(f"\nStart training CNN_LSTM + KDE(DW)... warm_up_epochs={warm_up_epochs}, SEQ_LEN={SEQ_LEN}, HORIZON={HORIZON}")
print(f"alpha_max(non-MF)={alpha_max}, alpha_mf={alpha_mf}, w_clip=[{w_clip_min},{w_clip_max}]")

for epoch in range(Freeze_epoch):
    model.train()
    train_loss_list = []

    # alpha warm-up（和你DW脚本一致）
    if warm_up_epochs > 0 and epoch < warm_up_epochs:
        alpha_main = alpha_max * (epoch + 1) / warm_up_epochs
        alpha_mf_now = alpha_mf * (epoch + 1) / warm_up_epochs
    else:
        alpha_main = alpha_max
        alpha_mf_now = alpha_mf

    alpha_vec = torch.full((num_groups,), float(alpha_main), dtype=torch.float32)
    alpha_vec[mf_id] = float(alpha_mf_now)

    # 计算 mean_w（窗口样本级）
    with torch.no_grad():
        alpha_per_sample = alpha_vec[train_gid_all].unsqueeze(1)  # (Nw,1)
        w_all = torch.clamp(1.0 - alpha_per_sample * train_p0_all, min=eps_dw)
        mean_w = w_all.mean()

    for xb, yb, p0b, gidb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        p0b = p0b.to(device)
        gidb = gidb.to(device)

        pred = model(xb)

        alpha_b = alpha_vec.to(device)[gidb].unsqueeze(1)  # (B,1)
        w_b = torch.clamp(1.0 - alpha_b * p0b, min=eps_dw) / (mean_w.to(device) + 1e-12)
        w_b = torch.clamp(w_b, w_clip_min, w_clip_max)

        lossvalue = loss_fn(pred, yb, w_b)

        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()

        train_loss_list.append(lossvalue.item())

    lr_scheduler.step()
    train_loss = float(np.mean(train_loss_list))

    rmse_all, rmse_mf, rmse_nonmf = eval_rmse_all_mf_nonmf(model, val_x, val_y_raw_cut)
    lr_now = optimizer.param_groups[0]['lr']

    print(
        f"Epoch {epoch+1}/{Freeze_epoch} | LR: {lr_now:.6f} | "
        f"alpha_main: {alpha_main:.3f} (MF {alpha_mf_now:.3f}) | "
        f"TrainLoss: {train_loss:.6f} | "
        f"ValRMSE(all): {rmse_all:.6f} | ValRMSE(MF): {rmse_mf:.6f} | ValRMSE(nonMF): {rmse_nonmf:.6f}"
    )

    # 每个 epoch 保存
    epoch_ckpt_path = os.path.join(
        base_ckpt_dir,
        f"CNNLSTM_DW_epoch_{epoch+1:03d}_rmseAll_{rmse_all:.6f}_rmseMF_{rmse_mf:.6f}.pth"
    )
    torch.save(model.state_dict(), epoch_ckpt_path)

    # best + early stop
    if rmse_all < best_val:
        best_val = rmse_all
        counter = 0

        if best_model_path is not None and os.path.exists(best_model_path):
            try:
                os.remove(best_model_path)
            except OSError:
                pass

        best_model_path = os.path.join(save_dir, f"CNNLSTM_DW_best_epoch_{epoch+1:03d}_rmseAll_{best_val:.6f}.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"  -> New best saved: {best_model_path}")
    else:
        if epoch >= warm_up_epochs:
            counter += 1
            print(f"  -> No improvement. Patience: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

print("\nTraining finished.")
print(f"Best Val RMSE(all): {best_val:.6f}")
print(f"Best model path: {best_model_path}")