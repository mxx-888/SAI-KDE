import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity

from SAI import MSMT_LE


def set_seed(seed=128):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(128)

x_cols = ['WS', 'Ta', 'RH', 'Press', 'SW_IN', 'LAI', 'SIF', 'acc_rain7', 'SM',
          'Month', 'DEM', 'GRA', 'WSA', 'SAV', 'EBF', 'DBF', 'MF', 'ENF',
          'WET', 'CRO', 'OSH', 'CSH', 'CLAY', 'SAND', 'SILT']
y_cols = ["ET", "H"]

LULC_COLS = ['GRA', 'WSA', 'SAV', 'EBF', 'DBF', 'MF', 'ENF', 'WET', 'CRO', 'OSH', 'CSH']
LULC_IDXS = [x_cols.index(c) for c in LULC_COLS]
num_groups = len(LULC_COLS)


def get_data(path):
    dfs = []
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        dfs.append(pd.read_csv(filepath))
    data = dfs[0]
    for i in range(1, len(dfs)):
        data = pd.concat([data, dfs[i]])
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


# ---------------- DenseWeight / DenseLoss  ----------------
def silverman_bandwidth(y_1d: np.ndarray):
    y = y_1d.astype(np.float64)
    n = len(y)
    std = np.std(y)
    h = 1.06 * std * (n ** (-1 / 5))
    return max(float(h), 1e-6)


def compute_p0_by_kde(y_1d: np.ndarray, bandwidth=None, kernel="gaussian"):
    y = y_1d.reshape(-1, 1).astype(np.float64)
    if bandwidth is None:
        bandwidth = silverman_bandwidth(y_1d)

    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kde.fit(y)
    log_p = kde.score_samples(y)
    p = np.exp(log_p)
    p0 = p / (p.max() + 1e-12)  # (0,1]
    return p0.reshape(-1, 1).astype(np.float32), float(bandwidth)


def compute_p0_by_kde_groupwise(y_raw, group_ids, num_groups, min_group_n=50):
    p0_all = np.zeros((len(y_raw), 1), dtype=np.float32)
    p0_global, global_bw = compute_p0_by_kde(y_raw, bandwidth=None)

    for g in range(num_groups):
        idx = (group_ids == g)
        if idx.sum() < min_group_n:
            p0_all[idx] = p0_global[idx]
        else:
            p0_g, _ = compute_p0_by_kde(y_raw[idx], bandwidth=None)
            p0_all[idx] = p0_g

    return p0_all, global_bw


# -----------------------------------------------------------------------------


Batchsize = 128
learning_rate = 0.01
Freeze_epoch = 300

path_train = r"D:\SAI\SAI\SAI-main\experiment\exp1\train"
path_val = r"D:\SAI\SAI\SAI-main\experiment\exp1\validation"
path_maxmin = r"D:\SAI\SAI\SAI-main\experiment\exp1\maxmin"

save_dir = r'D:\SAI\SAI\SAI-main\logs-dwtest-seed128'
os.makedirs(save_dir, exist_ok=True)

max_min_data = get_data(path_maxmin)
datamaxmin_x = np.array(max_min_data[x_cols]).astype(float)
maxx, minx, _ = data_norm(datamaxmin_x)

datamaxmin_y = np.array(max_min_data[y_cols]).astype(float)
maxy, miny, _ = data_norm(datamaxmin_y)

dataval = get_data(path_val)
datavalx = np.array(dataval[x_cols]).astype(float)
datavaly = np.array(dataval[y_cols]).astype(float)

for i in range(datavalx.shape[1]):
    if maxx[i] != minx[i]:
        datavalx[:, i:i + 1] = (datavalx[:, i:i + 1] - minx[i]) / (maxx[i] - minx[i])
    else:
        datavalx[:, i:i + 1] = 0.0

data = get_data(path_train)
datatrainx = np.array(data[x_cols]).astype(float)
datatrainy_raw = np.array(data[y_cols]).astype(float)
datatrainy = datatrainy_raw.copy()

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

group_ids = np.argmax(datatrainx[:, LULC_IDXS], axis=1).astype(np.int64)
counts = np.bincount(group_ids, minlength=num_groups)
print("[Group counts]", {LULC_COLS[i]: int(counts[i]) for i in range(num_groups)})

p0_np, global_bw = compute_p0_by_kde_groupwise(datatrainy_raw[:, 0], group_ids, num_groups, min_group_n=50)
print(f"[DenseWeight] Global KDE bandwidth = {global_bw:.6g}")
print("[DenseWeight] p0 quantiles:",
      np.quantile(p0_np, [0.01, 0.05, 0.5, 0.95, 0.99]).ravel())

# ======================= DataLoader ==========================
train_x = torch.from_numpy(datatrainx).float()
train_y = torch.from_numpy(datatrainy).float()
train_p0 = torch.from_numpy(p0_np).float()  # [N,1]

val_x = torch.from_numpy(datavalx).float()
val_y = torch.from_numpy(datavaly).float()

train_ds = Data.TensorDataset(train_x, train_y, train_p0)
train_loader = DataLoader(train_ds, batch_size=Batchsize, shuffle=True, num_workers=0)

# ======================= Loss ==========================
in_features = 9
num_heads = 3
hidden_features = 48
gate_num = 6
control_num = 16
gate_True = True
attention_layers = 10
linear_drop = 0.1


class MyLoss(nn.Module):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    @staticmethod
    def weighted_mean(per_sample_loss, w):
        return (per_sample_loss * w).sum() / (w.sum() + 1e-12)

    def forward(self, pred, y, w):
        w = w.to(pred.device)

        ET = y[:, 0:1];
        ET_pre = pred[:, 0:1]
        per_ori = (ET - ET_pre).pow(2)
        loss_ori = self.weighted_mean(per_ori, w)

        ET4 = ET.pow(4);
        ET4_pre = ET_pre.pow(4)
        per_add = (ET4 - ET4_pre).pow(2)
        loss_add = self.weighted_mean(per_add, w)

        H = y[:, 1:2];
        H_pre = pred[:, 1:2]
        loss_H = (H - H_pre).pow(2).mean()

        return self.a * loss_ori + (1 - self.a - self.c) * loss_add + self.c * loss_H


model = MSMT_LE(in_features, hidden_features, attention_layers, num_heads,
                gate_num, linear_drop, control_num, gate_True)

loss_fn = MyLoss(0.8, 0.1, 0.1)

warm_up_epochs = int(Freeze_epoch * 0.3)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
warm_up_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else \
    0.5 * (math.cos((epoch - warm_up_epochs) / (Freeze_epoch - warm_up_epochs) * math.pi) + 1)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warm_up_cosine_lr)

# ======================= DenseWeight  ==========================
alpha_max = 0.8
eps_dw = 1e-6
train_p0_all = train_p0  # [N,1]

patience = 50
best_val = float("inf")
counter = 0
best_model_path = None

print(f"Start training... Warm-up epochs: {warm_up_epochs}")

for epoch in range(Freeze_epoch):
    model.train()
    train_loss_list = []

    # alpha warm-up
    if warm_up_epochs > 0 and epoch < warm_up_epochs:
        alpha = alpha_max * (epoch + 1) / warm_up_epochs
    else:
        alpha = alpha_max

    with torch.no_grad():
        mean_w = torch.clamp(1.0 - alpha * train_p0_all, min=eps_dw).mean()

    for x_batch, y_batch, p0_batch in train_loader:
        pred = model(x_batch)

        w_batch = torch.clamp(1.0 - alpha * p0_batch, min=eps_dw) / (mean_w + 1e-12)

        lossvalue = loss_fn(pred, y_batch, w_batch)

        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()
        train_loss_list.append(lossvalue.item())

    lr_scheduler.step()
    train_loss = float(np.mean(train_loss_list))

    # validation
    model.eval()
    with torch.no_grad():
        val_pred = model(val_x)

    pred_et = val_pred[:, 0:1].detach().cpu().numpy()
    pred_et = pred_et * (maxy[0] - miny[0]) + miny[0]
    true_et = val_y[:, 0:1].detach().cpu().numpy()

    val_rmse = float(np.sqrt(mean_squared_error(pred_et, true_et)))

    q95 = np.quantile(true_et, 0.95)
    mask = (true_et[:, 0] >= q95)
    tail_rmse = float(np.sqrt(mean_squared_error(pred_et[mask], true_et[mask]))) if mask.any() else float('nan')

    lr_now = optimizer.param_groups[0]['lr']
    print(
        f"Epoch {epoch + 1}/{Freeze_epoch} | LR: {lr_now:.6f} | "
        f"alpha: {alpha:.3f} | Train Loss: {train_loss:.6f} | "
        f"Val RMSE: {val_rmse:.6f} | Val RMSE top5%: {tail_rmse:.6f}"
    )

    if val_rmse < best_val:
        best_val = val_rmse
        counter = 0

        if best_model_path is not None and os.path.exists(best_model_path):
            try:
                os.remove(best_model_path)
            except OSError:
                pass

        best_model_path = os.path.join(
            save_dir,
            f'Exp1DW-seed128-SAI_best_epoch_{epoch + 1}_val_{best_val:.6f}.pth'
        )
        torch.save(model.state_dict(), best_model_path)
        print(f"  -> New best model saved at: {best_model_path}")
    else:
        if epoch >= warm_up_epochs:
            counter += 1
            print(f"  -> No improvement. Patience: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

print("Training finished.")
print(f"Best val_RMSE_ET: {best_val:.6f}")
print(f"Final Best model location: {best_model_path}")