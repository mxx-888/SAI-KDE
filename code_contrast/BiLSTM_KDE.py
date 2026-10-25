# ======= 解决 Windows 下常见 OpenMP 重复加载崩溃（OMP Error #15）=======
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# =====================================================================================

import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error


# ======================= 固定随机种子 ==========================
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
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ======================= 列名 ==========================
FEATURE_COLS = [
    'WS','Ta','RH','Press','SW_IN','LAI','SIF','acc_rain7','SM','Month','DEM','GRA',
    'WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT'
]
TARGET_COLS = ["ET", "H"]

# LULC one-hot（用于分组KDE）
LULC_COLS = ['GRA', 'WSA', 'SAV', 'EBF', 'DBF', 'MF', 'ENF', 'WET', 'CRO', 'OSH', 'CSH']
LULC_IDXS = [FEATURE_COLS.index(c) for c in LULC_COLS]
num_groups = len(LULC_COLS)
mf_id = LULC_COLS.index("MF")


# ======================= 数据读取/归一化 ==========================
def get_data(path):
    dfs = []
    for file in os.listdir(path):
        fp = os.path.join(path, file)
        if os.path.isfile(fp) and file.lower().endswith(".csv"):
            dfs.append(pd.read_csv(fp))
    if len(dfs) == 0:
        raise FileNotFoundError(f"No csv files in: {path}")
    return pd.concat(dfs, ignore_index=True)

def data_norm(data):
    """返回每列max/min，并把data做min-max归一化（原地修改）。"""
    maxlist, minlist = [], []
    for i in range(data.shape[1]):
        col = data[:, i:i+1]
        mx = np.max(col)
        mn = np.min(col)
        maxlist.append(float(mx))
        minlist.append(float(mn))
        if mx != mn:
            data[:, i:i+1] = (col - mn) / (mx - mn)
        else:
            data[:, i:i+1] = 0.0
    return maxlist, minlist, data

def minmax_apply(X, min_list, max_list):
    X = X.copy().astype(np.float32)
    for i in range(X.shape[1]):
        mn, mx = min_list[i], max_list[i]
        den = (mx - mn)
        if den != 0:
            X[:, i:i+1] = (X[:, i:i+1] - mn) / den
        else:
            X[:, i:i+1] = 0.0
    return X


# ======================= 序列滑窗（给 BiLSTM 用） ==========================
def build_windows(X, Y, seq_len=30, step=0):
    """
    X: (N,F), Y:(N,2)
    返回:
      Xw: (Nwin, seq_len, F)
      Yw: (Nwin, 2) 取 i+seq_len-1+step
    """
    N = X.shape[0]
    max_i = N - (seq_len - 1) - step
    if max_i <= 0:
        return None, None
    Xw, Yw = [], []
    for i in range(max_i):
        Xw.append(X[i:i+seq_len, :])
        Yw.append(Y[i + seq_len - 1 + step, :])
    return np.asarray(Xw, dtype=np.float32), np.asarray(Yw, dtype=np.float32)

def load_folder_windows_train_dual(folder, seq_len, step, minx, maxx, miny, maxy):
    """
    训练用：每个CSV分别滑窗，避免跨文件串窗
    返回：
      Xw:      (Nwin,T,F)  X归一化
      Yw_norm: (Nwin,2)    y归一化（训练loss用）
      Yw_raw:  (Nwin,2)    y原始（KDE用）
    """
    all_Xw, all_Yw_norm, all_Yw_raw = [], [], []
    for file in os.listdir(folder):
        fp = os.path.join(folder, file)
        if (not os.path.isfile(fp)) or (not file.lower().endswith(".csv")):
            continue
        df = pd.read_csv(fp)
        X_raw = np.array(df[FEATURE_COLS]).astype(np.float32)
        Y_raw = np.array(df[TARGET_COLS]).astype(np.float32)

        Xn = minmax_apply(X_raw, minx, maxx)
        Yn = minmax_apply(Y_raw, miny, maxy)

        Xw, Yw_norm = build_windows(Xn, Yn, seq_len=seq_len, step=step)
        _,  Yw_raw  = build_windows(Xn, Y_raw, seq_len=seq_len, step=step)
        if Xw is None:
            continue

        all_Xw.append(Xw)
        all_Yw_norm.append(Yw_norm)
        all_Yw_raw.append(Yw_raw)

    if len(all_Xw) == 0:
        raise ValueError(f"No valid windows built from folder={folder}. Check seq_len/step or CSV length.")

    Xw = np.concatenate(all_Xw, axis=0)
    Yw_norm = np.concatenate(all_Yw_norm, axis=0)
    Yw_raw = np.concatenate(all_Yw_raw, axis=0)
    return Xw, Yw_norm, Yw_raw

def load_folder_windows(folder, seq_len, step, minx, maxx):
    """
    验证/测试：每个CSV分别滑窗，然后拼起来。
    返回：Xw_norm, Yw_raw
    """
    all_Xw, all_Yw = [], []
    for file in os.listdir(folder):
        fp = os.path.join(folder, file)
        if (not os.path.isfile(fp)) or (not file.lower().endswith(".csv")):
            continue
        df = pd.read_csv(fp)
        X_raw = np.array(df[FEATURE_COLS]).astype(np.float32)
        Y_raw = np.array(df[TARGET_COLS]).astype(np.float32)

        Xn = minmax_apply(X_raw, minx, maxx)
        Xw, Yw = build_windows(Xn, Y_raw, seq_len=seq_len, step=step)
        if Xw is None:
            continue
        all_Xw.append(Xw)
        all_Yw.append(Yw)

    if len(all_Xw) == 0:
        raise ValueError(f"No valid windows built from folder={folder}. Check seq_len/step or CSV length.")

    return np.concatenate(all_Xw, axis=0), np.concatenate(all_Yw, axis=0)


# ===================== DenseWeight: KDE Approx (No sklearn) =====================
def silverman_bandwidth(y_1d: np.ndarray):
    y = np.asarray(y_1d, dtype=np.float64).reshape(-1)
    n = len(y)
    std = np.std(y)
    h = 1.06 * std * (n ** (-1 / 5))
    return max(float(h), 1e-6)

def kde_1d_hist(y_1d, bins=512, bandwidth=None):
    """
    近似 KDE：直方图密度 + 高斯卷积平滑 + 插值回每个样本
    返回 p0 = p/p.max() in (0,1]
    """
    y = np.asarray(y_1d, dtype=np.float64).reshape(-1)
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
    """
    对每个LULC组分别做KDE；样本少的组用全局KDE代替
    """
    y_raw = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    group_ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)

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


# ======================= Loss：DW权重只作用于ET主项 ==========================
class MyLossDW(nn.Module):
    def __init__(self, a=0.8, c=0.1):
        super().__init__()
        self.a = a
        self.c = c

    @staticmethod
    def weighted_mean(per_sample_loss, w):
        return (per_sample_loss * w).sum() / (w.sum() + 1e-12)

    def forward(self, pred, y_norm, w=None):
        """
        pred: (B,2) 归一化空间输出
        y_norm: (B,2) 归一化空间标签
        w: (B,1) DenseWeight
        """
        ET = y_norm[:, 0:1]
        ET_pre = pred[:, 0:1]

        per_ori = (ET - ET_pre).pow(2)  # (B,1)
        if w is None:
            loss_ori = per_ori.mean()
        else:
            w = w.to(per_ori.device)
            loss_ori = self.weighted_mean(per_ori, w)

        ET4 = ET.pow(4)
        ET4_pre = ET_pre.pow(4)
        loss_add = (ET4 - ET4_pre).pow(2).mean()

        H = y_norm[:, 1:2]
        H_pre = pred[:, 1:2]
        loss_H = (H - H_pre).pow(2).mean()

        return self.a * loss_ori + (1 - self.a - self.c) * loss_add + self.c * loss_H


# ======================= BiLSTM 模型 ==========================
class BiLSTMRegressor(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_layers=2, dropout=0.0, out_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size * 2, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)      # (B,T,2H)
        last = out[:, -1, :]       # (B,2H)
        return self.fc(last)       # (B,2)


# ======================= 超参数 ==========================
Batchsize = 128
learning_rate = 0.01
Freeze_epoch = 300

SEQ_LEN = 30


LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.0

# DenseWeight(DW) 超参
USE_DW = True
alpha_max = 0.8     # non-MF
alpha_mf  = 0.3     # MF
eps_dw = 1e-6
w_clip_min, w_clip_max = 0.2, 2.0
kde_bins = 512
min_group_n = 50

path_train = r'.\experiment\exp1\train'
path_val   = r'.\experiment\exp1\validation'
path_maxmin = r'.\experiment\exp1\maxmin'

save_dir = r'.\logs-BiLSTM-DW'
os.makedirs(save_dir, exist_ok=True)

every_epoch_dir = os.path.join(save_dir, "every_epoch")
best_dir = os.path.join(save_dir, "best")
os.makedirs(every_epoch_dir, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)


# ======================= 计算 max/min（用 maxmin 文件夹） ==========================
max_min_data = get_data(path_maxmin)

X_mm = np.array(max_min_data[FEATURE_COLS]).astype(np.float32)
maxx, minx, _ = data_norm(X_mm)

Y_mm = np.array(max_min_data[TARGET_COLS]).astype(np.float32)
maxy, miny, _ = data_norm(Y_mm)

print("Loaded min/max.")


# ======================= 构造滑窗数据（训练：norm+raw；验证：raw） ==========================
train_Xw, train_Yw_norm, train_Yw_raw = load_folder_windows_train_dual(
    folder=path_train,
    seq_len=SEQ_LEN,
    step=PRED_STEP,
    minx=minx, maxx=maxx,
    miny=miny, maxy=maxy
)
val_Xw, val_Y_raw = load_folder_windows(
    folder=path_val,
    seq_len=SEQ_LEN,
    step=PRED_STEP,
    minx=minx, maxx=maxx
)

print("train windows:", train_Xw.shape, "val windows:", val_Xw.shape)

# window group_id（取每个window最后一步的LULC one-hot）
train_group_ids = np.argmax(train_Xw[:, -1, LULC_IDXS], axis=1).astype(np.int64)
val_group_ids   = np.argmax(val_Xw[:, -1, LULC_IDXS], axis=1).astype(np.int64)
val_mf_mask = (val_group_ids == mf_id)
val_nonmf_mask = ~val_mf_mask

counts = np.bincount(train_group_ids, minlength=num_groups)
print("[Train group counts]", {LULC_COLS[i]: int(counts[i]) for i in range(num_groups)})

# KDE -> p0（raw ET）
if USE_DW:
    p0_np, global_bw = compute_p0_by_kde_groupwise_hist(
        y_raw=train_Yw_raw[:, 0],          # raw ET
        group_ids=train_group_ids,
        num_groups=num_groups,
        min_group_n=min_group_n,
        bins=kde_bins
    )
    print(f"[DW] KDE bandwidth (global) = {global_bw:.6g}")
    print("[DW] p0 quantiles:", np.quantile(p0_np, [0.01, 0.05, 0.5, 0.95, 0.99]).ravel())
else:
    p0_np = np.zeros((train_Xw.shape[0], 1), dtype=np.float32)


# ======================= DataLoader ==========================
train_x = torch.from_numpy(train_Xw).float()
train_y = torch.from_numpy(train_Yw_norm).float()    # norm y
train_p0 = torch.from_numpy(p0_np).float()           # (N,1)
train_gid = torch.from_numpy(train_group_ids).long() # (N,)

val_x = torch.from_numpy(val_Xw).float().to(device)
val_y_raw = torch.from_numpy(val_Y_raw).float()      # raw on CPU ok

train_ds = Data.TensorDataset(train_x, train_y, train_p0, train_gid)
train_loader = DataLoader(train_ds, batch_size=Batchsize, shuffle=True, num_workers=0)

if len(train_ds) // Batchsize == 0:
    raise ValueError('数据集过小，无法训练，请扩充数据或减小Batchsize/SEQ_LEN。')


# ======================= 建模 ==========================
model = BiLSTMRegressor(
    num_features=len(FEATURE_COLS),
    hidden_size=LSTM_HIDDEN,
    num_layers=LSTM_LAYERS,
    dropout=LSTM_DROPOUT,
    out_dim=2
).to(device)

loss_fn = MyLossDW(a=0.8, c=0.1).to(device)


# ======================= Optimizer & LR ==========================
warm_up_epochs = int(Freeze_epoch * 0.3)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

warm_up_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else \
    0.5 * (math.cos((epoch - warm_up_epochs) / (Freeze_epoch - warm_up_epochs) * math.pi) + 1)

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warm_up_cosine_lr)


# ======================= 早停 + best ==========================
patience = 50
best_val = float('inf')
counter = 0
best_model_path = None

# 训练集全量张量（用于 mean_w）
train_p0_all = train_p0
train_gid_all = train_gid

print(f"Start training... USE_DW={USE_DW} | Warm-up epochs: {warm_up_epochs} | SEQ_LEN={SEQ_LEN}")

for epoch in range(Freeze_epoch):
    model.train()
    train_loss_list = []

    # alpha warm-up（nonMF 与 MF 同时 warm-up）
    if warm_up_epochs > 0 and epoch < warm_up_epochs:
        alpha_main = alpha_max * (epoch + 1) / warm_up_epochs
        alpha_mf_now = alpha_mf * (epoch + 1) / warm_up_epochs
    else:
        alpha_main = alpha_max
        alpha_mf_now = alpha_mf

    alpha_vec = torch.full((num_groups,), float(alpha_main), dtype=torch.float32)
    alpha_vec[mf_id] = float(alpha_mf_now)

    # mean_w（用全训练集计算）
    if USE_DW:
        with torch.no_grad():
            alpha_per_sample = alpha_vec[train_gid_all].unsqueeze(1)  # (N,1)
            w_all = torch.clamp(1.0 - alpha_per_sample * train_p0_all, min=eps_dw)
            mean_w = w_all.mean()
    else:
        mean_w = torch.tensor(1.0)

    for x_batch, y_batch, p0_batch, gid_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if USE_DW:
            p0_batch = p0_batch.to(device)
            gid_batch = gid_batch.to(device)

            alpha_batch = alpha_vec.to(device)[gid_batch].unsqueeze(1)  # (B,1)
            w_batch = torch.clamp(1.0 - alpha_batch * p0_batch, min=eps_dw) / (mean_w.to(device) + 1e-12)
            w_batch = torch.clamp(w_batch, w_clip_min, w_clip_max)
        else:
            w_batch = None

        pred = model(x_batch)
        lossvalue = loss_fn(pred, y_batch, w=w_batch)

        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()

        train_loss_list.append(lossvalue.item())

    lr_scheduler.step()
    train_loss = float(np.mean(train_loss_list))

    # ======================= 验证（RMSE算ET，raw尺度） ==========================
    model.eval()
    with torch.no_grad():
        val_pred_norm = model(val_x)  # (Nwin,2) norm

    pred_et_norm = val_pred_norm[:, 0].detach().cpu().numpy()
    pred_et = pred_et_norm * (maxy[0] - miny[0]) + miny[0]
    true_et = val_y_raw[:, 0].numpy()

    rmse_all = math.sqrt(mean_squared_error(pred_et, true_et))
    rmse_mf = math.sqrt(mean_squared_error(pred_et[val_mf_mask], true_et[val_mf_mask])) if val_mf_mask.any() else float("nan")
    rmse_nonmf = math.sqrt(mean_squared_error(pred_et[val_nonmf_mask], true_et[val_nonmf_mask])) if val_nonmf_mask.any() else float("nan")

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{Freeze_epoch} | LR: {current_lr:.6f} | "
          f"alpha_main={alpha_main:.3f}(MF {alpha_mf_now:.3f}) | "
          f"Train Loss: {train_loss:.6f} | Val RMSE(all): {rmse_all:.6f} | "
          f"RMSE(MF): {rmse_mf:.6f} | RMSE(nonMF): {rmse_nonmf:.6f}")

    # ======================= 每轮保存checkpoint（断点续训用） ==========================
    epoch_ckpt_path = os.path.join(
        every_epoch_dir,
        f"epoch_{epoch+1:03d}_rmseAll_{rmse_all:.6f}_rmseMF_{rmse_mf:.6f}.pth"
    )
    torch.save({
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": lr_scheduler.state_dict(),
        "best_val": best_val,
        "rmse_all": rmse_all,
        "rmse_mf": rmse_mf,
        "rmse_nonmf": rmse_nonmf,
        "use_dw": USE_DW,
        "alpha_main": float(alpha_main),
        "alpha_mf": float(alpha_mf_now),
        "seq_len": SEQ_LEN,
        "pred_step": PRED_STEP,
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "lulc_cols": LULC_COLS,
        "minx": minx, "maxx": maxx,
        "miny": miny, "maxy": maxy,
        "seed": 48,
    }, epoch_ckpt_path)

    # ======================= 保存best（只保留最新best） ==========================
    if rmse_all < best_val:
        best_val = rmse_all
        counter = 0

        if best_model_path is not None and os.path.exists(best_model_path):
            try:
                os.remove(best_model_path)
            except OSError:
                pass

        best_model_path = os.path.join(
            best_dir,
            f"BiLSTM_DW_best_epoch_{epoch+1:03d}_rmseAll_{best_val:.6f}_rmseMF_{rmse_mf:.6f}.pth"
        )
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": lr_scheduler.state_dict(),
            "best_val": best_val,
            "rmse_all": rmse_all,
            "rmse_mf": rmse_mf,
            "rmse_nonmf": rmse_nonmf,
            "use_dw": USE_DW,
            "alpha_main": float(alpha_main),
            "alpha_mf": float(alpha_mf_now),
            "seq_len": SEQ_LEN,
            "pred_step": PRED_STEP,
            "seed": 48,
        }, best_model_path)

        print(f"  -> New best saved: {best_model_path}")
    else:
        if epoch >= warm_up_epochs:
            counter += 1
            print(f"  -> No improvement. Patience: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

print("Training finished.")
print(f"Best val RMSE(all): {best_val:.6f}")
print(f"Every-epoch checkpoints: {every_epoch_dir}")
print(f"Best model: {best_model_path}")