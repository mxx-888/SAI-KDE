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
DIAG_ONLY = False          # True：只做诊断并退出；False：训练
TRAIN_BASE = True          # 训练 base 模型
TRAIN_ADAPTER = True       # 训练 MF-only adapter
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

LULC_COLS = ['GRA', 'WSA', 'SAV', 'EBF', 'DBF', 'MF', 'ENF', 'WET', 'CRO', 'OSH', 'CSH']
LULC_IDXS = [x_cols.index(c) for c in LULC_COLS]
num_groups = len(LULC_COLS)
mf_id = LULC_COLS.index("MF")
mf_feat_idx = x_cols.index("MF")

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


# ===================== Config =====================
Batchsize = 128
learning_rate = 0.01
Freeze_epoch = 300

alpha_max = 0.8
alpha_mf = 0.3
eps_dw = 1e-6
w_clip_min, w_clip_max = 0.2, 2.0

# ---- MF adapter config ----
adapter_hidden = 64
adapter_lr = 1e-3
adapter_wd = 1e-4
adapter_epochs = 30
adapter_patience = 8

path_train = r".\experiment\exp1\train"
path_val   = r".\experiment\exp1\validation"
path_maxmin = r".\experiment\exp1\maxmin"

save_dir = r".logs"
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
datatrainy = datatrainy_raw.copy()                     # will be normalized

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

# ===================== Group ids =====================
group_ids = np.argmax(datatrainx[:, LULC_IDXS], axis=1).astype(np.int64)
counts = np.bincount(group_ids, minlength=num_groups)
print("[Group counts]", {LULC_COLS[i]: int(counts[i]) for i in range(num_groups)})

val_group_ids = np.argmax(datavalx[:, LULC_IDXS], axis=1).astype(np.int64)
val_mf_mask = (val_group_ids == mf_id)
val_nonmf_mask = ~val_mf_mask

# ===================== DenseWeight p0 (groupwise) =====================
p0_np, global_bw = compute_p0_by_kde_groupwise_hist(
    datatrainy_raw[:, 0], group_ids, num_groups, min_group_n=50, bins=512
)
print(f"[DenseWeight] Global bandwidth (Silverman) = {global_bw:.6g}")
print("[DenseWeight] p0 quantiles:",
      np.quantile(p0_np, [0.01, 0.05, 0.5, 0.95, 0.99]).ravel())

# ===================== Diagnostics (optional) =====================
def diagnose_by_group(datatrainy_raw, group_ids, p0_np, LULC_COLS, alpha=0.8, eps=1e-6, save_csv=None):
    y = datatrainy_raw[:, 0].astype(np.float64)   # ET raw
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
            "ET_q10": np.quantile(y_g, 0.10),
            "ET_q50": np.quantile(y_g, 0.50),
            "ET_q90": np.quantile(y_g, 0.90),
            "ET_q95": np.quantile(y_g, 0.95),
            "p0_q10": np.quantile(p0_g, 0.10),
            "p0_q50": np.quantile(p0_g, 0.50),
            "p0_q90": np.quantile(p0_g, 0.90),
            "w_q10": np.quantile(w_g, 0.10),
            "w_q50": np.quantile(w_g, 0.50),
            "w_q90": np.quantile(w_g, 0.90),
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
    lulc_sum = datatrainx[:, LULC_IDXS].sum(axis=1)
    print("\n===== LULC one-hot check =====")
    print("sum quantiles:", np.quantile(lulc_sum, [0, 0.01, 0.5, 0.99, 1.0]))
    print("sum==1:", np.sum(np.isclose(lulc_sum, 1.0)))
    print("sum==0:", np.sum(np.isclose(lulc_sum, 0.0)))
    print("sum>1:", np.sum(lulc_sum > 1.0 + 1e-6))

    diagnose_by_group(
        datatrainy_raw, group_ids, p0_np, LULC_COLS,
        alpha=alpha_max, save_csv=os.path.join(save_dir, "diag_group_stats.csv")
    )
    print("\nDIAG_ONLY=True -> exit before training.")
    sys.exit(0)

# ======================= DataLoader (加入 group_id) ==========================
train_x = torch.from_numpy(datatrainx).float()
train_y = torch.from_numpy(datatrainy).float()      # normalized y
train_p0 = torch.from_numpy(p0_np).float()          # [N,1]
train_gid = torch.from_numpy(group_ids).long()      # [N]

val_x = torch.from_numpy(datavalx).float().to(device)
val_y_raw = torch.from_numpy(datavaly).float()      # raw on CPU is ok

train_ds = Data.TensorDataset(train_x, train_y, train_p0, train_gid)
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
train_gid_all = train_gid  # CPU tensor ok


def eval_rmse_all_mf_nonmf(model_, val_x_, val_y_raw_):
    model_.eval()
    with torch.no_grad():
        val_pred_norm = model_(val_x_)  # (N,2) normalized

    pred_et = val_pred_norm[:, 0:1].detach().cpu().numpy()
    pred_et = pred_et * (maxy[0] - miny[0]) + miny[0]  # denorm to raw
    true_et = val_y_raw_[:, 0:1].detach().cpu().numpy()

    rmse_all = rmse_np(pred_et, true_et)
    rmse_mf = rmse_np(pred_et[val_mf_mask], true_et[val_mf_mask]) if val_mf_mask.any() else float("nan")
    rmse_nonmf = rmse_np(pred_et[val_nonmf_mask], true_et[val_nonmf_mask]) if val_nonmf_mask.any() else float("nan")
    return rmse_all, rmse_mf, rmse_nonmf

base_ckpt_dir = os.path.join(save_dir, "base_every_epoch")
os.makedirs(base_ckpt_dir, exist_ok=True)
if TRAIN_BASE:
    print(f"Start BASE training... Warm-up epochs: {warm_up_epochs}")
    print(f"alpha_max(non-MF)={alpha_max}, alpha_mf={alpha_mf}, w_clip=[{w_clip_min},{w_clip_max}]")

    for epoch in range(Freeze_epoch):
        model.train()
        train_loss_list = []

        # ========= A 修改：MF alpha 也 warm-up，并且真的写入 alpha_vec =========
        if warm_up_epochs > 0 and epoch < warm_up_epochs:
            alpha_main = alpha_max * (epoch + 1) / warm_up_epochs
            alpha_mf_now = alpha_mf * (epoch + 1) / warm_up_epochs
        else:
            alpha_main = alpha_max
            alpha_mf_now = alpha_mf

        alpha_vec = torch.full((num_groups,), float(alpha_main), dtype=torch.float32)
        alpha_vec[mf_id] = float(alpha_mf_now)
        # ====================================================================

        with torch.no_grad():
            alpha_per_sample = alpha_vec[train_gid_all].unsqueeze(1)  # [N,1]
            w_all = torch.clamp(1.0 - alpha_per_sample * train_p0_all, min=eps_dw)
            mean_w = w_all.mean()

        for x_batch, y_batch, p0_batch, gid_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            p0_batch = p0_batch.to(device)
            gid_batch = gid_batch.to(device)

            pred = model(x_batch)

            alpha_batch = alpha_vec.to(device)[gid_batch].unsqueeze(1)  # [B,1]
            w_batch = torch.clamp(1.0 - alpha_batch * p0_batch, min=eps_dw) / (mean_w.to(device) + 1e-12)
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
        print(
            f"[BASE] Epoch {epoch + 1}/{Freeze_epoch} | LR: {lr_now:.6f} | "
            f"alpha_main: {alpha_main:.3f} (MF {alpha_mf_now:.3f}) | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val RMSE(all): {rmse_all:.6f} | Val RMSE(MF): {rmse_mf:.6f} | Val RMSE(nonMF): {rmse_nonmf:.6f}"
        )
        # ===== 每个 epoch 都保存 =====
        epoch_ckpt_path = os.path.join(
            base_ckpt_dir,
            f"BASE_epoch_{epoch + 1:03d}_rmseAll_{rmse_all:.6f}_rmseMF_{rmse_mf:.6f}.pth"
        )
        torch.save(model.state_dict(), epoch_ckpt_path)
        print(f"  -> Saved BASE checkpoint: {epoch_ckpt_path}")
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
                print(f"  -> No improvement. Patience: {counter}/{patience}")
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

    print("BASE training finished.")
    print(f"Best BASE val_RMSE_ET(all): {best_val:.6f}")
    print(f"Final Best BASE model location: {best_model_path}")


# ======================= MF-only Residual Adapter ==========================
class MFResidualAdapter(nn.Module):
    def __init__(self, x_dim, hidden=64, mf_feat_idx=0):
        super().__init__()
        self.mf_feat_idx = mf_feat_idx
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)   # ET residual
        )

    def forward(self, x):
        mf_gate = (x[:, self.mf_feat_idx:self.mf_feat_idx+1] > 0.5).float()
        return self.net(x) * mf_gate

adapter_ckpt_dir = os.path.join(save_dir, "adapter_every_epoch")
os.makedirs(adapter_ckpt_dir, exist_ok=True)
if TRAIN_ADAPTER:
    if best_model_path is None or (not os.path.exists(best_model_path)):
        raise RuntimeError("best_model_path is None or not found. Please train BASE first or set correct path.")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    x_dim = train_x.shape[1]
    adapter = MFResidualAdapter(x_dim=x_dim, hidden=adapter_hidden, mf_feat_idx=mf_feat_idx).to(device)

    mf_train_mask = (train_gid.numpy() == mf_id)
    if mf_train_mask.sum() == 0:
        raise RuntimeError("No MF samples in training set.")

    train_x_mf = train_x[mf_train_mask]
    train_y_mf = train_y[mf_train_mask]  # normalized
    mf_ds = Data.TensorDataset(train_x_mf, train_y_mf)
    mf_loader = DataLoader(mf_ds, batch_size=Batchsize, shuffle=True, num_workers=0)

    adapter_opt = torch.optim.AdamW(adapter.parameters(), lr=adapter_lr, weight_decay=adapter_wd)

    best_mf_rmse = float("inf")
    bad = 0
    best_adapter_path = os.path.join(save_dir, "MF_adapter_best.pth")

    print("\n===================== Start MF-ADAPTER finetune =====================")
    print(f"MF samples: {int(mf_train_mask.sum())} | adapter_hidden={adapter_hidden} | lr={adapter_lr}")

    for ep in range(adapter_epochs):
        adapter.train()
        loss_list = []

        for xb, yb in mf_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            with torch.no_grad():
                base_pred = model(xb)              # normalized (B,2)

            delta_et = adapter(xb)                 # (B,1)
            pred_et = base_pred[:, 0:1] + delta_et

            loss = (pred_et - yb[:, 0:1]).pow(2).mean()

            adapter_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            adapter_opt.step()

            loss_list.append(loss.item())

        adapter.eval()
        model.eval()
        with torch.no_grad():
            base_val = model(val_x)               # (N,2) normalized
            delta_val = adapter(val_x)            # (N,1) nonMF=0
            fused_val = base_val.clone()
            fused_val[:, 0:1] = base_val[:, 0:1] + delta_val

        pred_et = fused_val[:, 0:1].detach().cpu().numpy()
        pred_et_raw = pred_et * (maxy[0] - miny[0]) + miny[0]
        true_et_raw = val_y_raw[:, 0:1].detach().cpu().numpy()

        rmse_all = rmse_np(pred_et_raw, true_et_raw)
        rmse_mf = rmse_np(pred_et_raw[val_mf_mask], true_et_raw[val_mf_mask]) if val_mf_mask.any() else float("nan")
        rmse_nonmf = rmse_np(pred_et_raw[val_nonmf_mask], true_et_raw[val_nonmf_mask]) if val_nonmf_mask.any() else float("nan")

        with torch.no_grad():
            base_et_raw = (base_val[:, 0:1].detach().cpu().numpy() * (maxy[0] - miny[0]) + miny[0])
        max_abs_diff_nonmf = np.max(np.abs(pred_et_raw[val_nonmf_mask] - base_et_raw[val_nonmf_mask])) if val_nonmf_mask.any() else 0.0

        print(
            f"[ADAPTER] epoch {ep+1}/{adapter_epochs} | train_loss={np.mean(loss_list):.6f} | "
            f"Val RMSE(all)={rmse_all:.6f} | Val RMSE(MF)={rmse_mf:.6f} | Val RMSE(nonMF)={rmse_nonmf:.6f} | "
            f"max|diff| nonMF={max_abs_diff_nonmf:.3e}"
        )
        # ===== 每个 epoch 都保存 adapter =====
        adapter_epoch_path = os.path.join(
            adapter_ckpt_dir,
            f"ADAPTER_epoch_{ep + 1:03d}_rmseAll_{rmse_all:.6f}_rmseMF_{rmse_mf:.6f}.pth"
        )
        torch.save(adapter.state_dict(), adapter_epoch_path)
        print(f"  -> Saved ADAPTER checkpoint: {adapter_epoch_path}")
        if rmse_mf < best_mf_rmse:
            best_mf_rmse = rmse_mf
            bad = 0
            torch.save(adapter.state_dict(), best_adapter_path)
            print(f"  -> New best ADAPTER saved: {best_adapter_path}")
        else:
            bad += 1
            if bad >= adapter_patience:
                print("  -> MF-adapter early stopping.")
                break

    print("===================== finished =====================")
    print(f"Best Val RMSE(MF) after adapter: {best_mf_rmse:.6f}")
    print(f"Best adapter location: {best_adapter_path}")