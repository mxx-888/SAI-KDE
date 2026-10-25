# ======================= RNN + KDE(DenseWeight) 训练脚本（只训练，不含预测输出文件） =======================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ======================= Feature / Label =======================
FEATURE_ORDER = [
    'WS', 'Ta', 'RH', 'Press', 'SW_IN', 'LAI', 'SIF', 'acc_rain7', 'SM', 'Month', 'DEM', 'GRA', 'WSA',
    'SAV', 'EBF', 'DBF', 'MF', 'ENF', 'WET', 'CRO', 'OSH', 'CSH', 'CLAY', 'SAND', 'SILT'
]
Y_COLS = ["ET", "H"]
INPUT_DIM = len(FEATURE_ORDER)

TIME_COL_CANDIDATES = ["TIMESTAMP", "timestamp", "date", "Date", "time", "Time"]

LULC_COLS = ['GRA', 'WSA', 'SAV', 'EBF', 'DBF', 'MF', 'ENF', 'WET', 'CRO', 'OSH', 'CSH']
LULC_IDXS = [FEATURE_ORDER.index(c) for c in LULC_COLS]
num_groups = len(LULC_COLS)
mf_id = LULC_COLS.index("MF")

# ======================= Path =======================
path_train  = r".\experiment\exp1\train"
path_val    = r".\experiment\exp1\validation"
path_maxmin = r".\experiment\exp1\maxmin"

save_dir = r".\logs-Rnn"
os.makedirs(save_dir, exist_ok=True)
ckpt_dir = os.path.join(save_dir, "every_epoch")
os.makedirs(ckpt_dir, exist_ok=True)

# ===================== Config =====================
Batchsize = 128
learning_rate = 0.01
epochs = 300

alpha_max = 0.8
alpha_mf = 0.3
eps_dw = 1e-6
w_clip_min, w_clip_max = 0.2, 2.0

warmup_ratio = 0.1
warm_up_epochs = int(epochs * warmup_ratio)

patience = 20

# KDE config（你原代码里用到但没定义）
use_groupwise_kde = True
min_group_n = 50
kde_bins = 512

# 序列长度（你原代码里用到 seq_len，但没定义）
seq_len = 30

# ---- MF adapter config (aligned) ----
# （这段目前在脚本里没用到，保留不影响；如果不用可删）
adapter_hidden = 64
adapter_lr = learning_rate
adapter_wd = 1e-4
adapter_epochs = epochs
adapter_patience = 8
adapter_batchsize = Batchsize

# 为了避免变量名混用，统一一下训练用的名称
batch_size = Batchsize
lr = learning_rate

# ======================= IO / Norm =======================
def get_data_with_site(path_dir):
    dfs = []
    for fn in sorted(os.listdir(path_dir)):
        fp = os.path.join(path_dir, fn)
        if os.path.isfile(fp) and fn.lower().endswith(".csv"):
            df = pd.read_csv(fp)
            df["site_id"] = os.path.splitext(fn)[0]
            dfs.append(df)
    if len(dfs) == 0:
        raise FileNotFoundError(f"No csv files in: {path_dir}")
    return pd.concat(dfs, axis=0, ignore_index=True)

def get_minmax_from_maxmin_dir(path_dir, cols):
    df = get_data_with_site(path_dir)
    arr = df[cols].to_numpy(dtype=np.float64)
    col_max = np.nanmax(arr, axis=0)
    col_min = np.nanmin(arr, axis=0)
    return col_max, col_min

maxx, minx = get_minmax_from_maxmin_dir(path_maxmin, FEATURE_ORDER)
maxy, miny = get_minmax_from_maxmin_dir(path_maxmin, Y_COLS)

def minmax_norm(arr, col_min, col_max, clip01=True):
    arr = np.asarray(arr, dtype=np.float64)
    out = arr.copy()
    for i in range(out.shape[1]):
        mx, mn = col_max[i], col_min[i]
        den = mx - mn
        if den != 0:
            out[:, i] = (out[:, i] - mn) / den
        else:
            out[:, i] = 0.0
    if clip01:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)

def minmax_denorm(arr_norm, col_min, col_max):
    arr_norm = np.asarray(arr_norm, dtype=np.float64)
    out = arr_norm.copy()
    for i in range(out.shape[1]):
        mx, mn = col_max[i], col_min[i]
        out[:, i] = out[:, i] * (mx - mn) + mn
    return out

# ======================= KDE（近似：直方图 + 高斯卷积） ==========================
def silverman_bandwidth(y_1d: np.ndarray):
    y = np.asarray(y_1d, dtype=np.float64).reshape(-1)
    n = len(y)
    std = np.std(y)
    h = 1.06 * std * (n ** (-1 / 5))
    return max(float(h), 1e-6)

def kde_1d_hist(y_1d, bins=512, bandwidth=None):
    y = np.asarray(y_1d, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return np.zeros((0, 1), dtype=np.float32), 1e-6

    nan_mask = np.isnan(y)
    if nan_mask.any():
        y = y.copy()
        y[nan_mask] = np.nanmedian(y)

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

# ======================= Build sequences ==========================
def build_sequences_from_df(df, seq_len=30):
    miss = [c for c in (FEATURE_ORDER + Y_COLS + ["site_id"]) if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns: {miss}")

    time_col = None
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            time_col = c
            break

    X_seq_list, Y_list, Yraw_list, gid_list = [], [], [], []

    for site, g in df.groupby("site_id"):
        g = g.copy()
        if time_col is not None:
            g = g.sort_values(time_col)

        x_raw = g[FEATURE_ORDER].to_numpy(dtype=np.float32)
        y_raw = g[Y_COLS].to_numpy(dtype=np.float32)

        x = minmax_norm(x_raw, minx, maxx, clip01=True)
        y = minmax_norm(y_raw, miny, maxy, clip01=True)

        L = x.shape[0]
        if L < seq_len:
            continue

        for t in range(seq_len - 1, L):
            X_seq_list.append(x[t - seq_len + 1: t + 1, :])
            Y_list.append(y[t])
            Yraw_list.append(y_raw[t])

            lulc_vec = x_raw[t, LULC_IDXS]  # 用 raw 的 LULC 判别组别
            gid_list.append(int(np.argmax(lulc_vec)))

    X_seq = np.stack(X_seq_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    Y_raw = np.stack(Yraw_list, axis=0).astype(np.float32)
    gid = np.asarray(gid_list, dtype=np.int64)
    return X_seq, Y, Y_raw, gid

class SeqDatasetDW(Data.Dataset):
    def __init__(self, X_seq, Y, p0, gid):
        self.X_seq = torch.from_numpy(X_seq).float()
        self.Y = torch.from_numpy(Y).float()
        self.p0 = torch.from_numpy(p0).float()
        self.gid = torch.from_numpy(gid).long()

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X_seq[idx], self.Y[idx], self.p0[idx], self.gid[idx]

# ======================= RNN 模型 ==========================
class RNN_ET_25(nn.Module):
    def __init__(self, input_dim=25, hidden=64, num_layers=2, dropout=0.1, out_dim=2, nonlinearity="tanh"):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x_seq):
        _, h_n = self.rnn(x_seq)
        h_last = h_n[-1]
        return self.head(h_last)

# ======================= DenseWeight Loss（只加权 ET） ==========================
class DenseWeightLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def weighted_mean(per_sample_loss, w):
        return (per_sample_loss * w).sum() / (w.sum() + 1e-12)

    def forward(self, pred, y, w):
        w = w.to(pred.device)
        et_err2 = (pred[:, 0:1] - y[:, 0:1]).pow(2)
        loss_et = self.weighted_mean(et_err2, w)
        loss_h = (pred[:, 1:2] - y[:, 1:2]).pow(2).mean()
        return loss_et + loss_h

# ======================= 指标工具 ==========================
def rmse_np(pred, true):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((pred - true) ** 2)))

# ======================= main ==========================
def main():
    # 读数据
    df_train = get_data_with_site(path_train)
    df_val = get_data_with_site(path_val)

    Xtr_seq, Ytr, Ytr_raw, gid_tr = build_sequences_from_df(df_train, seq_len=seq_len)
    Xva_seq, Yva, Yva_raw, gid_va = build_sequences_from_df(df_val, seq_len=seq_len)

    print("Train seq samples:", Xtr_seq.shape, Ytr.shape, "gid:", gid_tr.shape)
    print("Val   seq samples:", Xva_seq.shape, Yva.shape, "gid:", gid_va.shape)

    # ---- KDE: 计算 p0（用训练集 ET raw + gid）----
    if use_groupwise_kde:
        p0_tr, global_bw = compute_p0_by_kde_groupwise_hist(
            y_raw=Ytr_raw[:, 0],
            group_ids=gid_tr,
            num_groups=num_groups,
            min_group_n=min_group_n,
            bins=kde_bins
        )
        print(f"[KDE] groupwise enabled | global_bw={global_bw:.6g} | bins={kde_bins}")
    else:
        p0_tr, global_bw = kde_1d_hist(Ytr_raw[:, 0], bins=kde_bins, bandwidth=None)
        print(f"[KDE] global only | bw={global_bw:.6g} | bins={kde_bins}")

    print("[KDE] p0 quantiles:", np.quantile(p0_tr, [0.01, 0.05, 0.5, 0.95, 0.99]).ravel())
    counts = np.bincount(gid_tr, minlength=num_groups)
    print("[Group counts]", {LULC_COLS[i]: int(counts[i]) for i in range(num_groups)})

    # ---- DataLoader ----
    train_ds = SeqDatasetDW(Xtr_seq, Ytr, p0_tr, gid_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    val_x_seq = torch.from_numpy(Xva_seq).float().to(device)
    val_y_raw = Yva_raw  # numpy raw

    # ---- 模型/优化器/损失 ----
    model = RNN_ET_25(input_dim=INPUT_DIM, hidden=64, num_layers=2, dropout=0.1, out_dim=2, nonlinearity="tanh").to(device)
    loss_fn = DenseWeightLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- 早停 & 保存 ----
    best_rmse_et = float("inf")
    best_path = None
    bad = 0

    # 训练集全量 p0/gid（算 mean_w 用）
    train_p0_all = torch.from_numpy(p0_tr).float()      # (N,1) CPU
    train_gid_all = torch.from_numpy(gid_tr).long()     # (N,) CPU

    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        # ===== alpha warmup（并给 MF 单独 alpha）=====
        if warm_up_epochs > 0 and ep <= warm_up_epochs:
            alpha_main = alpha_max * ep / warm_up_epochs
            alpha_mf_now = alpha_mf * ep / warm_up_epochs
        else:
            alpha_main = alpha_max
            alpha_mf_now = alpha_mf

        alpha_vec = torch.full((num_groups,), float(alpha_main), dtype=torch.float32)
        alpha_vec[mf_id] = float(alpha_mf_now)

        # ===== mean_w：用全体训练样本统计，保持 batch 内归一 =====
        with torch.no_grad():
            alpha_per_sample = alpha_vec[train_gid_all].unsqueeze(1)  # (N,1)
            w_all = torch.clamp(1.0 - alpha_per_sample * train_p0_all, min=eps_dw)
            mean_w = w_all.mean()  # CPU tensor

        for xb_seq, yb, p0b, gidb in train_loader:
            xb_seq = xb_seq.to(device)
            yb = yb.to(device)
            p0b = p0b.to(device)
            gidb = gidb.to(device)

            pred = model(xb_seq)

            alpha_batch = alpha_vec.to(device)[gidb].unsqueeze(1)  # (B,1)
            w_batch = torch.clamp(1.0 - alpha_batch * p0b, min=eps_dw) / (mean_w.to(device) + 1e-12)
            w_batch = torch.clamp(w_batch, w_clip_min, w_clip_max)

            loss = loss_fn(pred, yb, w_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

        # ===== 验证：ET/H RMSE（原始尺度）=====
        model.eval()
        with torch.no_grad():
            val_pred_norm = model(val_x_seq).detach().cpu().numpy()  # (N,2) norm

        val_pred_raw = minmax_denorm(val_pred_norm, miny, maxy)      # (N,2) raw
        rmse_et = rmse_np(val_pred_raw[:, 0], val_y_raw[:, 0])
        rmse_h  = rmse_np(val_pred_raw[:, 1], val_y_raw[:, 1])

        cur_lr = optimizer.param_groups[0]["lr"]
        train_loss = float(np.mean(losses))

        # 每个 epoch 保存一次
        epoch_path = os.path.join(ckpt_dir, f"rnnKDE_epoch_{ep:03d}_rmseET_{rmse_et:.6f}.pth")
        torch.save(model.state_dict(), epoch_path)

        print(
            f"Epoch {ep:03d}/{epochs} | lr={cur_lr:.6g} | "
            f"train_loss={train_loss:.6f} | alpha_main={alpha_main:.3f}(MF {alpha_mf_now:.3f}) | "
            f"val_RMSE_ET={rmse_et:.6f} | val_RMSE_H={rmse_h:.6f}"
        )

        # 早停：保存 best
        if rmse_et < best_rmse_et:
            best_rmse_et = rmse_et
            bad = 0

            if best_path is not None and os.path.exists(best_path):
                try:
                    os.remove(best_path)
                except OSError:
                    pass

            best_path = os.path.join(save_dir, f"rnnKDE_best_rmseET_{best_rmse_et:.6f}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best saved: {best_path}")
        else:
            if ep > warm_up_epochs:
                bad += 1
                print(f"  -> No improvement. Patience: {bad}/{patience}")
                if bad >= patience:
                    print("Early stopping.")
                    break

    print("\nTraining finished.")
    print("Best val RMSE(ET):", best_rmse_et)
    print("Best model path:", best_path)

if __name__ == "__main__":
    main()