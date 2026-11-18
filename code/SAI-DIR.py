import os
import math
import random
import functools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from KDEpy import FFTKDE
from SAI_new import MSMT_LE_TOKEN


def bisection(array, value):
    """Locate value within a monotonically increasing array."""
    n = len(array)
    if value < array[0]:
        return -1
    if value > array[n - 1]:
        return n

    jl, ju = 0, n - 1
    while ju - jl > 1:
        jm = (ju + jl) >> 1
        if value >= array[jm]:
            jl = jm
        else:
            ju = jm

    if value == array[0]:
        return 0
    if value == array[n - 1]:
        return n - 1
    return jl


class TargetRelevance:
    """DenseWeight from Steininger et al. [23]."""

    def __init__(self, y, alpha=1.0):
        self.alpha = alpha
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        print("TargetRelevance alpha:", self.alpha)

        silverman_bandwidth = 1.06 * np.std(y) * np.power(len(y), -1.0 / 5.0)
        print("Using Silverman Bandwidth", silverman_bandwidth)

        self.kernel = FFTKDE(bw=silverman_bandwidth).fit(y, weights=None)
        x, y_dens_grid = self.kernel.evaluate(4096)
        self.x = x

        # Scale the density grid to [0, 1]; a PDF value may exceed 1.
        self.y_dens_grid = MinMaxScaler().fit_transform(
            y_dens_grid.reshape(-1, 1)
        ).flatten()
        self.y_dens = np.vectorize(self.get_density)(y)

        self.eps = 1e-6
        w_star = np.maximum(1 - self.alpha * self.y_dens, self.eps)
        self.mean_w_star = np.mean(w_star)
        self.relevances = w_star / self.mean_w_star

    def get_density(self, y):
        idx = bisection(self.x, y)
        try:
            return self.y_dens_grid[idx]
        except IndexError:
            if idx <= -1:
                idx = 0
            elif idx >= len(self.x):
                idx = len(self.x) - 1
            return self.y_dens_grid[idx]

    @functools.lru_cache(maxsize=100000)
    def eval_single(self, y):
        dens = self.get_density(y)
        return np.maximum(1 - self.alpha * dens, self.eps) / self.mean_w_star

    def eval(self, y):
        ys = np.asarray(y).flatten().tolist()
        return np.array(list(map(self.eval_single, ys)))[:, None]

    def __call__(self, y):
        return self.eval(y)


# ======================= 固定随机种子 =======================
SEED = 48

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# "none": SAI 基线；"denseweight": SAI + 文献[23] DenseWeight
WEIGHTING = os.environ.get("WEIGHTING", "denseweight")
DW_ALPHA = float(os.environ.get("DW_ALPHA", 1.0))


def get_data(path):
    data_list = []
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        data_list.append(pd.read_csv(filepath))
    data = data_list[0]
    for i in range(1, len(data_list)):
        data = pd.concat([data, data_list[i]])
    return data


def data_norm(data):
    maxlist, minlist = [], []
    for i in range(data.shape[1]):
        input_data = data[:, i:i + 1]
        max_v = np.max(input_data)
        min_v = np.min(input_data)
        maxlist.append(max_v)
        minlist.append(min_v)
        data[:, i:i + 1] = (input_data - min_v) / (max_v - min_v)
    return maxlist, minlist, data


Batchsize = 128
learning_rate = 0.01
Freeze_epoch = 300

path = r"D:\SAI\SAI\SAI-main\experiment\exp1\train"
test_val = r"D:\SAI\SAI\SAI-main\experiment\exp1\validation"
FEATURES = [
    "WS", "Ta", "RH", "Press", "SW_IN", "LAI", "SIF", "acc_rain7", "SM",
    "Month", "DEM", "GRA", "WSA", "SAV", "EBF", "DBF", "MF", "ENF", "WET",
    "CRO", "OSH", "CSH", "CLAY", "SAND", "SILT",
]

max_min_data = get_data(r"D:\SAI\SAI\SAI-main\experiment\exp1\maxmin")
datamaxmin = np.array(max_min_data[FEATURES]).astype(float)
maxx, minx, _ = data_norm(datamaxmin)
print(maxx)
print("==========")
print(minx)
print("==========")

datamaxminy = np.array(max_min_data[["ET", "H"]]).astype(float)
maxy, miny, _ = data_norm(datamaxminy)
print(maxy)
print("==========")
print(miny)
print("==========")

dataval = get_data(test_val)
datavalx = np.array(dataval[FEATURES]).astype(float)
datavaly = np.array(dataval[["ET", "H"]]).astype(float)
for i in range(datavalx.shape[1]):
    datavalx[:, i:i + 1] = (datavalx[:, i:i + 1] - minx[i]) / (maxx[i] - minx[i])

data = get_data(path)
datatrainx = np.array(data[FEATURES]).astype(float)
datatrainy = np.array(data[["ET", "H"]]).astype(float)
for i in range(datatrainx.shape[1]):
    if maxx[i] != minx[i]:
        datatrainx[:, i:i + 1] = (
            (datatrainx[:, i:i + 1] - minx[i]) / (maxx[i] - minx[i])
        )
for i in range(datatrainy.shape[1]):
    if maxy[i] != miny[i]:
        datatrainy[:, i:i + 1] = (
            (datatrainy[:, i:i + 1] - miny[i]) / (maxy[i] - miny[i])
        )

num_train = datatrainx.shape[0]
print(num_train)

train_x = torch.from_numpy(datatrainx).float()
train_y = torch.from_numpy(datatrainy).float()
validationx_data = torch.from_numpy(datavalx).float()
validationy_data = torch.from_numpy(datavaly).float()

# 权重只由训练集 ET 计算，验证集不参与
if WEIGHTING == "denseweight":
    et_train = datatrainy[:, 0]
    tr = TargetRelevance(et_train, alpha=DW_ALPHA)
    w_np = np.asarray(tr.relevances, dtype=np.float32).reshape(-1)

    bw_silverman = 1.06 * np.std(et_train) * len(et_train) ** (-1.0 / 5.0)
    floor_frac = float(np.mean((1 - DW_ALPHA * tr.y_dens) <= tr.eps))
    print(f"[DenseWeight] alpha={DW_ALPHA}, bandwidth(Silverman)={bw_silverman:.6f}")
    print(
        f"[DenseWeight] weight min/max/mean/std = "
        f"{w_np.min():.4f}/{w_np.max():.4f}/{w_np.mean():.4f}/{w_np.std():.4f}"
    )
    print(f"[DenseWeight] fraction of samples clipped to eps: {floor_frac:.4f}")
elif WEIGHTING == "none":
    w_np = np.ones(num_train, dtype=np.float32)
    bw_silverman = None
    floor_frac = None
else:
    raise ValueError(f"Unknown WEIGHTING: {WEIGHTING}")

sample_weights = torch.from_numpy(w_np)
train_loader = DataLoader(
    Data.TensorDataset(train_x, train_y, sample_weights),
    batch_size=Batchsize,
    shuffle=True,
    num_workers=0,
)

in_features = 9
num_heads = 3
hidden_features = 48
gate_num = 6
control_num = 16
gate_True = True
attention_layers = 10
linear_drop = 0.1


class MyLossWeighted(nn.Module):
    def __init__(self, a, c):
        super().__init__()
        self.a = a
        self.c = c

    def forward(self, Pre, Test, w):
        w = w.view(-1, 1)
        ET, ET_pre = Test[:, 0:1], Pre[:, 0:1]
        H, H_pre = Test[:, 1:2], Pre[:, 1:2]

        loss_ori = torch.mean(w * torch.pow(ET - ET_pre, 2))
        loss_add = torch.mean(
            w * torch.pow(torch.pow(ET, 4) - torch.pow(ET_pre, 4), 2)
        )
        loss_H = torch.mean(
            torch.pow(H - H_pre, 2)
        )
        return self.a * loss_ori + (1 - self.a - self.c) * loss_add + self.c * loss_H


modeltrain = MSMT_LE_TOKEN(
    in_features, hidden_features, attention_layers, num_heads,
    gate_num, linear_drop, control_num, gate_True
).train()
loss = MyLossWeighted(0.8, 0.1)

if num_train // Batchsize == 0:
    raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

# AdamW + cosine annealing (no warm-up)
optimizer = torch.optim.AdamW(
    modeltrain.parameters(),
    lr=learning_rate,
    weight_decay=1e-4,
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=Freeze_epoch,
    eta_min=1e-6,
)

patience = 50
best_val = float("inf")
counter = 0
best_model_path = None
save_dir = r"D:\SAI\SAI\SAI-main\logs-all\SAI-DIR"
os.makedirs(save_dir, exist_ok=True)

print(
    f"Start training... WEIGHTING={WEIGHTING}, "
    f"DW_ALPHA={DW_ALPHA}, Optimizer=AdamW, "
    f"Initial LR={learning_rate}, Weight Decay=1e-4, "
    f"Scheduler=CosineAnnealingLR, Epochs={Freeze_epoch}"
)

for epoch in range(Freeze_epoch):
    modeltrain.train()
    train_loss_list = []

    for x_train, y_train, w_train in train_loader:
        output_train = modeltrain(x_train)
        lossvalue = loss(output_train, y_train, w_train)
        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()
        train_loss_list.append(lossvalue.item())

    lr_scheduler.step()
    train_loss = float(np.mean(train_loss_list))

    modeltrain.eval()
    with torch.no_grad():
        validation_output = modeltrain(validationx_data)

    val = math.sqrt(mean_squared_error(
        np.array(validation_output)[:, 0:1] * (maxy[0] - miny[0]) + miny[0],
        validationy_data[:, 0:1],
    ))

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch + 1}/{Freeze_epoch} | LR: {current_lr:.6f} | "
        f"Train Loss: {train_loss:.6f} | Val RMSE: {val:.6f}"
    )

    # 保存当前轮模型
    epoch_model_path = os.path.join(
        save_dir,
        f"Exp1seed{SEED}-{WEIGHTING}-a{DW_ALPHA}-"
        f"SAI_epoch_{epoch + 1:03d}_val_{val:.6f}.pth",
    )
    torch.save(modeltrain.state_dict(), epoch_model_path)
    print(f"  -> Epoch model saved at: {epoch_model_path}")

    # 保留原有最佳模型保存与早停逻辑
    if val < best_val:
        best_val = val
        counter = 0
        if best_model_path is not None and os.path.exists(best_model_path):
            try:
                os.remove(best_model_path)
            except OSError:
                pass

        best_model_path = os.path.join(
            save_dir,
            f"Exp1seed{SEED}-{WEIGHTING}-a{DW_ALPHA}-"
            f"SAI_best_epoch_{epoch + 1}_val_{best_val:.6f}.pth",
        )
        torch.save(modeltrain.state_dict(), best_model_path)
        print(f"  -> New best model saved at: {best_model_path}")
print("Training finished.")
print(f"Best val_RMSE_ET: {best_val:.6f}")
print(f"Final Best model location: {best_model_path}")

if WEIGHTING == "denseweight":
    print("=" * 60)
    print("DenseWeight configuration:")
    print(f"  alpha = {DW_ALPHA}, eps = 1e-6")
    print(f"  Silverman bandwidth = {bw_silverman:.6f}")
    print("  FFTKDE grid = 4096; out-of-range values use boundary density")
    print(f"  clipped-to-eps fraction = {floor_frac:.4f}")
    print("=" * 60)
