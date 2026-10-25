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

from SAI import MSMT_LE

# ===================== Switch =====================
DIAG_ONLY = False  # True：只做诊断并退出；False：训练
TRAIN_BASE = True  # 训练 base 模型


# <--- 修改/删除：移除了 TRAIN_ADAPTER 开关，因为该阶段已被删除
# TRAIN_ADAPTER = False
# ==================================================


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

# 保留LULC列用于评估，但不再用于训练加权
LULC_COLS = ['GRA', 'WSA', 'SAV', 'EBF', 'DBF', 'MF', 'ENF', 'WET', 'CRO', 'OSH', 'CSH']
LULC_IDXS = [x_cols.index(c) for c in LULC_COLS]
num_groups = len(LULC_COLS)
mf_id = LULC_COLS.index("MF")


# ===================== IO / Norm =====================
def get_data(path):
    dfs = []
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        if os.path.isfile(filepath) and file.lower().endswith(".csv"):
            dfs.append(pd.read_csv(filepath))
    if len(dfs) == 0:
        raise FileNotFoundError(f"No csv files found in: {path}")
    data = pd.concat(dfs, axis=0, ignore_index=True)
    return data


def data_norm(data):
    maxlist, minlist = [], []
    for i in range(data.shape[1]):
        col = data[:, i:i + 1]
        mx = np.max(col)
        mn = np.min(col)
        maxlist.append(mx)
        minlist.append(mn)
        if mx != mn:
            data[:, i:i + 1] = (col - mn) / (mx - mn)
        else:
            data[:, i:i + 1] = 0.0
    return maxlist, minlist, data


def rmse_np(pred, true):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


# ===================== DenseWeight: KDE Approx (No sklearn) =====================
def silverman_bandwidth(y_1d: np.ndarray):
    y = np.asarray(y_1d, dtype=np.float64)
    n = len(y)
    std = np.std(y)
    h = 1.06 * std * (n ** (-1 / 5))
    return max(float(h), 1e-6)


def kde_1d_hist(y_1d, bins=512, bandwidth=None):
    """
    近似 KDE：直方图密度 + 高斯卷积平滑 + 插值回每个样本
    返回 p0 = p/p.max() in (0,1]
    """
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


# <--- 修改/删除：删除了分组计算密度的函数 `compute_p0_by_kde_groupwise_hist`


# ===================== Config =====================
Batchsize = 128
learning_rate = 0.01
Freeze_epoch = 300

alpha_max = 0.8
# <--- 修改/删除：不再需要 alpha_mf
# alpha_mf = 0.3
eps_dw = 1e-6
w_clip_min, w_clip_max = 0.2, 2.0

# <--- 修改/删除：不再需要 adapter 相关配置
# adapter_hidden = 64
# adapter_lr = 1e-3
# adapter_wd = 1e-4
# adapter_epochs = 30
# adapter_patience = 8

path_train = r"D:\SAI\SAI\SAI-main\experiment\exp1\train"
path_val = r"D:\SAI\SAI\SAI-main\experiment\exp1\validation"
path_maxmin = r"D:\SAI\SAI\SAI-main\experiment\exp1\maxmin"

save_dir = r"D:\SAI\SAI\SAI-main\logs-dwtest-global-seed48"  # <--- 修改：更改了保存目录以作区分
os.makedirs(save_dir, exist_ok=True)

# ===================== Load / Normalize =====================
max_min_data = get_data(path_maxmin)
datamaxmin_x = np.array(max_min_data[x_cols]).astype(float)
maxx, minx, _ = data_norm(datamaxmin_x)

datamaxmin_y = np.array(max_min_data[y_cols]).astype(float)
maxy, miny, _ = data_norm(datamaxmin_y)

dataval = get_data(path_val)
datavalx = np.array(dataval[x_cols]).astype(float)
datavaly = np.array(dataval[y_cols]).astype(float)  # raw

for i in range(datavalx.shape[1]):
    if maxx[i] != minx[i]:
        datavalx[:, i:i + 1] = (datavalx[:, i:i + 1] - minx[i]) / (maxx[i] - minx[i])
    else:
        datavalx[:, i:i + 1] = 0.0

data = get_data(path_train)
datatrainx = np.array(data[x_cols]).astype(float)
datatrainy_raw = np.array(data[y_cols]).astype(float)  # raw
datatrainy = datatrainy_raw.copy()  # will be normalized

for i in range(datatrainx.shape[1]):
    if maxx[i] != minx[i]:
        datatrainx[:, i:i + 1] = (datatrainx[:, i:i + 1] - minx[i]) / (maxx[i] - minx[i])
    else:
        datatrainx[:, i:i + 1] = 0.0

for i in range(datatrainy.shape[1]):
    if maxy[i] != miny[i]:
        datatrainy[:, i:i + 1] = (datatrainy[:, i:i + 1] - miny[i]) / (maxy[i] - miny[i])
    else:
        datatrainy[:, i:i + 1] = 0.0

num_train = datatrainx.shape[0]
print("num_train:", num_train)

# ===================== Group ids (仅用于评估) =====================
val_group_ids = np.argmax(datavalx[:, LULC_IDXS], axis=1).astype(np.int64)
val_mf_mask = (val_group_ids == mf_id)
val_nonmf_mask = ~val_mf_mask

# ===================== DenseWeight p0 (Global) =====================
# <--- 修改：直接使用 kde_1d_hist 计算全局密度，不再分组
p0_np, global_bw = kde_1d_hist(
    datatrainy_raw[:, 0], bins=512
)
print(f"[DenseWeight] Global bandwidth (Silverman) = {global_bw:.6g}")
print("[DenseWeight] p0 quantiles (Global):",
      np.quantile(p0_np, [0.01, 0.05, 0.5, 0.95, 0.99]).ravel())


# ===================== Diagnostics (optional) =====================
# 诊断函数仍然保留，但现在它分析的是全局加权在不同组上的表现
def diagnose_by_group(datatrainy_raw, group_ids, p0_np, LULC_COLS, alpha=0.8, eps=1e-6, save_csv=None):
    y = datatrainy_raw[:, 0].astype(np.float64)  # ET raw
    p0 = p0_np[:, 0].astype(np.float64)

    w = np.clip(1.0 - alpha * p0, eps, None)
    w_global_mean = w.mean()

    rows = []
    for g, name in enumerate(LULC_COLS):
        idx = (group_ids == g)
        n = int(idx.sum())
        if n == 0:
            continue
        y_g, p0_g, w_g = y[idx], p0[idx], w[idx]
        rows.append({
            "group": name, "n": n,
            "ET_q10": np.quantile(y_g, 0.10), "ET_q50": np.quantile(y_g, 0.50), "ET_q90": np.quantile(y_g, 0.90),
            "ET_q95": np.quantile(y_g, 0.95),
            "p0_q10": np.quantile(p0_g, 0.10), "p0_q50": np.quantile(p0_g, 0.50), "p0_q90": np.quantile(p0_g, 0.90),
            "w_q10": np.quantile(w_g, 0.10), "w_q50": np.quantile(w_g, 0.50), "w_q90": np.quantile(w_g, 0.90),
            "w_mean": float(np.mean(w_g)),
            "w_mean_over_global": float(np.mean(w_g) / (w_global_mean + 1e-12)),
        })

    df = pd.DataFrame(rows).sort_values("n", ascending=False)
    print("\n===== Group diagnostics (ET / p0 / weight) =====")
    print(df.to_string(index=False))

    if save_csv is not None:
        df.to_csv(save_csv, index=False)
        print(f"\nSaved diagnostics to: {save_csv}")
    return df


if DIAG_ONLY:
    # 诊断部分需要 group_ids，所以在这里计算
    group_ids_diag = np.argmax(datatrainx[:, LULC_IDXS], axis=1).astype(np.int64)
    diagnose_by_group(
        datatrainy_raw, group_ids_diag, p0_np, LULC_COLS,
        alpha=alpha_max, save_csv=os.path.join(save_dir, "diag_group_stats_global.csv")
    )
    print("\nDIAG_ONLY=True -> exit before training.")
    sys.exit(0)

# ======================= DataLoader (不再需要 group_id) ==========================
train_x = torch.from_numpy(datatrainx).float()
train_y = torch.from_numpy(datatrainy).float()  # normalized y
train_p0 = torch.from_numpy(p0_np).float()  # [N,1]

val_x = torch.from_numpy(datavalx).float().to(device)
val_y_raw = torch.from_numpy(datavaly).float()  # raw on CPU is ok

# <--- 修改：TensorDataset不再包含 group_id
train_ds = Data.TensorDataset(train_x, train_y, train_p0)
train_loader = DataLoader(train_ds, batch_size=Batchsize, shuffle=True, num_workers=0)

# ======================= Model / Loss ==========================
in_features = 9
num_heads = 3
hidden_features = 48
gate_num = 6
control_num = 16
gate_True = True
attention_layers = 10
linear_drop = 0.1


class MyLoss(nn.Module):
    def __init__(self, a, c):
        super().__init__()
        self.a = a
        self.c = c

    @staticmethod
    def weighted_mean(per_sample_loss, w):
        return (per_sample_loss * w).sum() / (w.sum() + 1e-12)

    def forward(self, pred, y, w):
        w = w.to(pred.device)
        ET = y[:, 0:1]
        ET_pre = pred[:, 0:1]
        per_ori = (ET - ET_pre).pow(2)
        loss_ori = self.weighted_mean(per_ori, w)
        ET4 = ET.pow(4)
        ET4_pre = ET_pre.pow(4)
        per_add = (ET4 - ET4_pre).pow(2)
        loss_add = per_add.mean()
        H = y[:, 1:2]
        H_pre = pred[:, 1:2]
        loss_H = (H - H_pre).pow(2).mean()
        return self.a * loss_ori + (1 - self.a - self.c) * loss_add + self.c * loss_H


model = MSMT_LE(in_features, hidden_features, attention_layers, num_heads,
                gate_num, linear_drop, control_num, gate_True).to(device)

loss_fn = MyLoss(a=0.8, c=0.1)
warm_up_epochs = int(Freeze_epoch * 0.3)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
warm_up_cosine_lr = lambda epoch: (
    epoch / warm_up_epochs if epoch <= warm_up_epochs else
    0.5 * (math.cos((epoch - warm_up_epochs) / (Freeze_epoch - warm_up_epochs) * math.pi) + 1)
)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warm_up_cosine_lr)

# ======================= Training (base) ==========================
patience = 50
best_val = float("inf")
counter = 0
best_model_path = None

train_p0_all = train_p0  # CPU tensor ok


def eval_rmse_all_mf_nonmf(model_, val_x_, val_y_raw_):
    model_.eval()
    with torch.no_grad():
        val_pred_norm = model_(val_x_)
    pred_et = val_pred_norm[:, 0:1].detach().cpu().numpy()
    pred_et = pred_et * (maxy[0] - miny[0]) + miny[0]
    true_et = val_y_raw_[:, 0:1].detach().cpu().numpy()
    rmse_all = rmse_np(pred_et, true_et)
    rmse_mf = rmse_np(pred_et[val_mf_mask], true_et[val_mf_mask]) if val_mf_mask.any() else float("nan")
    rmse_nonmf = rmse_np(pred_et[val_nonmf_mask], true_et[val_nonmf_mask]) if val_nonmf_mask.any() else float("nan")
    return rmse_all, rmse_mf, rmse_nonmf


base_ckpt_dir = os.path.join(save_dir, "base_every_epoch")
os.makedirs(base_ckpt_dir, exist_ok=True)
if TRAIN_BASE:
    print(f"Start BASE training... Warm-up epochs: {warm_up_epochs}")
    # <--- 修改：打印信息简化，只显示一个 alpha
    print(f"alpha_global={alpha_max}, w_clip=[{w_clip_min},{w_clip_max}]")

    for epoch in range(Freeze_epoch):
        model.train()
        train_loss_list = []

        # <--- 修改：alpha 不再分组，只有一个全局值
        if warm_up_epochs > 0 and epoch < warm_up_epochs:
            alpha_now = alpha_max * (epoch + 1) / warm_up_epochs
        else:
            alpha_now = alpha_max

        # <--- 修改：mean_w 的计算也相应简化
        with torch.no_grad():
            w_all = torch.clamp(1.0 - alpha_now * train_p0_all, min=eps_dw)
            mean_w = w_all.mean()

        # <--- 修改：dataloader 不再返回 gid_batch
        for x_batch, y_batch, p0_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            p0_batch = p0_batch.to(device)

            pred = model(x_batch)

            # <--- 修改：权重计算使用全局的 alpha_now，不再需要 alpha_batch
            w_batch = torch.clamp(1.0 - alpha_now * p0_batch, min=eps_dw) / (mean_w.to(device) + 1e-12)
            w_batch = torch.clamp(w_batch, w_clip_min, w_clip_max)

            lossvalue = loss_fn(pred, y_batch, w_batch)

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            train_loss_list.append(lossvalue.item())

        lr_scheduler.step()
        train_loss = float(np.mean(train_loss_list))
        rmse_all, rmse_mf, rmse_nonmf = eval_rmse_all_mf_nonmf(model, val_x, val_y_raw)
        lr_now = optimizer.param_groups[0]['lr']

        # <--- 修改：打印信息简化
        print(
            f"[BASE] Epoch {epoch + 1}/{Freeze_epoch} | LR: {lr_now:.6f} | "
            f"alpha_global: {alpha_now:.3f} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val RMSE(all): {rmse_all:.6f} | Val RMSE(MF): {rmse_mf:.6f} | Val RMSE(nonMF): {rmse_nonmf:.6f}"
        )

        epoch_ckpt_path = os.path.join(
            base_ckpt_dir,
            f"BASE_epoch_{epoch + 1:03d}_rmseAll_{rmse_all:.6f}_rmseMF_{rmse_mf:.6f}.pth"
        )
        torch.save(model.state_dict(), epoch_ckpt_path)

        if rmse_all < best_val:
            best_val = rmse_all
            counter = 0
            if best_model_path is not None and os.path.exists(best_model_path):
                try:
                    os.remove(best_model_path)
                except OSError:
                    pass
            best_model_path = os.path.join(
                save_dir,
                f'BASE_best_epoch_{epoch + 1}_val_{best_val:.6f}.pth'
            )
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best BASE model saved at: {best_model_path}")
        else:
            if epoch >= warm_up_epochs:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

    print("BASE training finished.")
    print(f"Best BASE val_RMSE_ET(all): {best_val:.6f}")
    print(f"Final Best BASE model location: {best_model_path}")

