import torch
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from SAI_new import MSMT_LE_TOKEN
import math
import os
from sklearn.metrics import mean_squared_error
import random
from collections import Counter              # ★ 新增
from scipy.ndimage import convolve1d         # ★ 新增

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

def get_data(path):
    list = []
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        filedata = pd.read_csv(filepath)
        list.append(filedata)
    data = list[0]
    for i in range(len(list)):
        if i == 0:
            continue
        data = pd.concat([data, list[i]])
    return data

def data_norm(data):
    maxlist = []
    minlist = []
    for i in range(data.shape[1]):
        input_data = data[:, i:i+1]
        max = np.max(input_data)
        min = np.min(input_data)
        maxlist.append(max)
        minlist.append(min)
        input_data = (input_data - min) / (max - min)
        data[:, i:i+1] = input_data
    return maxlist, minlist, data

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# ★ ================================================================
# ★ 新增：DIR-LDS 基线的全部超参数（每一项都需写入论文/回复审稿人）
# ★ 这些超参应用验证集选择，且搜索预算须与其他基线对齐
# ★ ================================================================
WEIGHTING     = 'lds'     # 'none' -> SAI 基线（不加权）；'lds' -> SAI + DIR(LDS)
NB_BINS       = 100       # 直方图分箱数（归一化 ET [0,1] 上等宽分箱）
LDS_KERNEL    = 'gaussian'# 核类型
LDS_KS        = 5         # 核大小（奇数）
LDS_SIGMA     = 2         # 核带宽 sigma
REWEIGHT      = 'inverse'# 重加权方式：'sqrt_inv'（DIR 默认）或 'inverse'
EFF_NUM_FLOOR = 1.0       # 有效样本数下限（数值稳定，防除零）
NORM_WEIGHTS  = True      # 权重归一化到均值 1，保持损失量级与基线一致
                          # （DIR 官方实现未做此步，论文中需如实说明该适配）

def get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2):
    """
    ★ 新增：LDS 平滑核窗口（对应 DIR 论文中的高斯核平滑，峰在中心）。
    如需与 DIR 官方仓库逐位一致，可将本函数替换为
    imdb-wiki-dir/utils.py 中的原版 get_lds_kernel_window，
    并在论文中报告最终采用的版本。
    """
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    x = np.arange(ks) - half_ks
    if kernel == 'gaussian':
        base_kernel = np.exp(-0.5 * (x / sigma) ** 2)
    elif kernel == 'triang':
        base_kernel = np.maximum(0.0, 1.0 - np.abs(x) / (half_ks + 1))
    else:  # laplace
        base_kernel = np.exp(-np.abs(x) / sigma)
    return base_kernel / base_kernel.sum()

def compute_lds_sample_weights(et_labels):
    """
    ★ 新增：仅用训练集标签计算每个样本的 LDS 权重（无信息泄漏）。
    流程与 DIR 一致：分箱 -> 经验分布 -> 核平滑得有效分布 -> 逆有效样本数加权。
    """
    et = np.asarray(et_labels, dtype=np.float64)
    # 1) 分箱
    bin_idx = np.clip((et * NB_BINS).astype(int), 0, NB_BINS - 1)
    # 2) 经验标签分布
    num_of_bins = dict(Counter(bin_idx.tolist()))
    emp_dist = np.array([num_of_bins.get(i, 0) for i in range(NB_BINS)], dtype=np.float64)
    # 3) 高斯核平滑（边界处理：constant，即零填充）
    kernel_window = get_lds_kernel_window(kernel=LDS_KERNEL, ks=LDS_KS, sigma=LDS_SIGMA)
    eff_dist = convolve1d(emp_dist, weights=kernel_window, mode='constant')
    # 4) 每个样本的有效样本数 -> 权重
    eff_num = np.maximum(eff_dist[bin_idx], EFF_NUM_FLOOR)
    if REWEIGHT == 'sqrt_inv':
        w = 1.0 / np.sqrt(eff_num)
    elif REWEIGHT == 'inverse':
        w = 1.0 / eff_num
    else:
        raise ValueError(f"Unknown REWEIGHT: {REWEIGHT}")
    # 5) 可选均值归一化
    if NORM_WEIGHTS:
        w = w / max(float(w.mean()), 1e-12)

    # 6) 最终裁剪（裁剪后均值可能不再精确为 1）
    MAX_WEIGHT = 5.0
    MIN_WEIGHT = 0.1
    w = np.clip(w, MIN_WEIGHT, MAX_WEIGHT)

    # 7) 返回权重 Tensor，修复 NoneType 报错
    return torch.from_numpy(w.astype(np.float32))

train_ratio = 0.9
Batchsize = 128
learning_rate = 0.01
Freeze_epoch = 300

path = r'D:\SAI\SAI\SAI-main\experiment\exp1\train'
test_val = r'D:\SAI\SAI\SAI-main\experiment\exp1\validation'

max_min_data = get_data(r'D:\SAI\SAI\SAI-main\experiment\exp1\maxmin')
datamaxmin = np.array(max_min_data[['WS','Ta','RH','Press','SW_IN','LAI','SIF','acc_rain7','SM','Month','DEM','GRA',
                                    'WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float)

maxx,minx,_ = data_norm(datamaxmin)
print(maxx)
print('==========')
print(minx)
print('==========')
datamaxminy = np.array(max_min_data[["ET",'H']]).astype(float)
maxy,miny,_ = data_norm(datamaxminy)
print(maxy)
print('==========')
print(miny)
print('==========')

dataval = get_data(test_val)
datavalx = np.array(dataval[['WS','Ta','RH','Press','SW_IN','LAI','SIF','acc_rain7','SM','Month','DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float)
datavaly = np.array(dataval[["ET",'H']]).astype(float)
for i in range(datavalx.shape[1]):
    input_data = datavalx[:, i:i + 1]
    input_data = (input_data - minx[i]) / (maxx[i] - minx[i])
    datavalx[:, i:i + 1] = input_data

data = get_data(path)
datatrainx = np.array(data[['WS','Ta','RH','Press','SW_IN','LAI','SIF','acc_rain7','SM','Month','DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float)
datatrainy = np.array(data[["ET",'H']]).astype(float)
for i in range(datatrainx.shape[1]):
    input_data = datatrainx[:, i:i + 1]
    if maxx[i] != minx[i]:
        input_data = (input_data - minx[i]) / (maxx[i] - minx[i])
    datatrainx[:, i:i + 1] = input_data
for i in range(datatrainy.shape[1]):
    input_data = datatrainy[:, i:i + 1]
    if maxy[i] != miny[i]:
        input_data = (input_data - miny[i]) / (maxy[i] - miny[i])
    datatrainy[:, i:i + 1] = input_data

num = datatrainx.shape[0]
print(num)

trainx_data = datatrainx
trainy_data = datatrainy
num_train = len(datatrainx)
train_x = torch.from_numpy(trainx_data).float()
train_y = torch.from_numpy(trainy_data).float()
validationx_data = torch.from_numpy(datavalx).float()
validationy_data = torch.from_numpy(datavaly).float()

# ★ ================================================================
# ★ 新增：计算训练集样本权重（仅用训练集 ET 标签，避免信息泄漏）
# ★ ================================================================
if WEIGHTING == 'lds':
    # 以归一化 ET（[0,1]）作为加权的标签目标，与 DIR 用目标值分箱的做法一致
    sample_weights = compute_lds_sample_weights(datatrainy[:, 0])
    w_np = sample_weights.numpy()
    print(f"[LDS] weight stats: min={w_np.min():.4f}, max={w_np.max():.4f}, "
          f"mean={w_np.mean():.4f}, std={w_np.std():.4f}")
    print(f"[LDS] config: bins={NB_BINS}, kernel={LDS_KERNEL}, ks={LDS_KS}, "
          f"sigma={LDS_SIGMA}, reweight={REWEIGHT}, floor={EFF_NUM_FLOOR}, "
          f"norm_mean1={NORM_WEIGHTS}")
else:
    # SAI 基线：权重全 1，与原始不加权损失严格等价
    sample_weights = torch.ones(num, dtype=torch.float32)

# ★ 修改：Dataset 同时返回权重
train_torch_dataset = Data.TensorDataset(train_x, train_y, sample_weights)
train_loader = DataLoader(train_torch_dataset, batch_size=Batchsize, shuffle=True, num_workers=0)

in_features = 9
sparameters = 7
num_heads = 3
hidden_features = 48
parameter_features = 20
gate_num = 6
control_num = 16
gate_True = True
attention_layers = 10
linear_drop = 0.1

# ★ 修改：加权版损失。加权约定与 DIR 的 weighted_mse_loss 一致：mean(w * err^2)。
# ★ 当 w 全为 1（WEIGHTING='none'）时，与原始 MyLoss 完全等价。
# ★ 系数保持不变：0.8*ori + 0.1*add + 0.1*H
class MyLoss_weighted(nn.Module):
    def __init__(self, a, c):
        super(MyLoss_weighted, self).__init__()
        self.a = a
        self.c = c
    def forward(self, Pre, Test, w):
        w = w.view(-1, 1)

        ET = Test[:, 0:1]
        ET_pre = Pre[:, 0:1]
        loss_ori = torch.mean(w * torch.pow(ET - ET_pre, 2))

        ET_low = torch.pow(ET, 4)
        ET_low_pre = torch.pow(ET_pre, 4)
        loss_add = torch.mean(w * torch.pow(ET_low - ET_low_pre, 2))

        H = Test[:, 1:2]
        H_pre = Pre[:, 1:2]
        loss_H = torch.mean(w * torch.pow(H - H_pre, 2))

        loss = self.a * loss_ori + (1 - self.a - self.c) * loss_add + self.c * loss_H
        return loss

model = MSMT_LE_TOKEN(in_features, hidden_features, attention_layers, num_heads, gate_num, linear_drop, control_num, gate_True)
modeltrain = model.train()
loss = MyLoss_weighted(0.8, 0.1)   # ★ 系数与原 MyLoss(0.8, 0.1, 0.1) 等价
epoch_step = num_train // Batchsize

model_path = r""
if model_path != '':
    print('Loading weights into state dict...')
    model_dict = modeltrain.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    modeltrain.load_state_dict(model_dict)

if epoch_step == 0:
    raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

warm_up_epochs = int(Freeze_epoch * 0.3)

learning_rate = 0.01

optimizer = torch.optim.AdamW(
    modeltrain.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=Freeze_epoch,
    eta_min=1e-6
)
patience = 50
best_val = float('inf')
counter = 0
best_model_path = None
save_dir = r'D:\SAI\SAI\SAI-main\logs-all\SAI-LDS'
os.makedirs(save_dir, exist_ok=True)

print(f"Start training... WEIGHTING={WEIGHTING}, Warm-up epochs: {warm_up_epochs}")

for epoch in range(Freeze_epoch):
    modeltrain.train()
    train_loss_list = []

    # ★ 修改：解包权重
    for step, (x_train, y_train, w_train) in enumerate(train_loader):
        output_train = modeltrain(x_train)
        lossvalue = loss(output_train, y_train, w_train)
        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()
        train_loss_list.append(lossvalue.item())

    lr_scheduler.step()
    train_loss = float(np.mean(train_loss_list))

    # 验证（不加权，与基线口径一致）
    modeltrain.eval()
    with torch.no_grad():
        validation_output = modeltrain(validationx_data)

    val = math.sqrt(mean_squared_error(
        (np.array(validation_output)[:, 0:1]) * (maxy[0] - miny[0]) + miny[0],
        validationy_data[:, 0:1]
    ))

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}/{Freeze_epoch} | LR: {current_lr:.6f} | "
          f"Train Loss: {train_loss:.6f} | Val RMSE: {val:.6f}")

    # 每一轮都保存模型
    epoch_model_path = os.path.join(
        save_dir,
        f'Exp1seed48-{WEIGHTING}-SAI_epoch_{epoch + 1:03d}_val_{val:.6f}.pth'
    )
    torch.save(modeltrain.state_dict(), epoch_model_path)
    print(f"  -> Epoch model saved at: {epoch_model_path}")

    # 原有的最佳模型保存与早停逻辑
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
            f'Exp1seed48-{WEIGHTING}-SAI_best_epoch_{epoch + 1}_val_{best_val:.6f}.pth'
        )
        torch.save(modeltrain.state_dict(), best_model_path)
        print(f"  -> New best model saved at: {best_model_path}")
    else:
        if epoch >= warm_up_epochs:
            counter += 1
            print(f"  -> No improvement. Patience: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        else:
            pass

print(f"Training finished.")
print(f"Best val_RMSE_ET: {best_val:.6f}")
print(f"Final Best model location: {best_model_path}")

# ★ 新增：训练结束后打印配置摘要，直接用于论文与审稿回复
if WEIGHTING == 'lds':
    print("=" * 60)
    print("DIR-LDS baseline configuration (for paper / rebuttal):")
    print(f"  - Weighting: LDS + {REWEIGHT} re-weighting")
    print(f"  - Histogram bins: {NB_BINS} (equal-width on normalized ET in [0,1])")
    print(f"  - Kernel: {LDS_KERNEL}, ks={LDS_KS}, sigma={LDS_SIGMA}, window sum-normalized")
    print(f"  - convolve1d boundary mode: constant (zero padding)")
    print(f"  - Effective-count floor: {EFF_NUM_FLOOR}")
    print(f"  - Weight normalization: mean-1 over training set = {NORM_WEIGHTS}")
    print(f"  - Weight applied to: full per-sample loss (ET + ET^4 + H terms)")
    print(f"  - LDS warm-up: none (static weights); early-stopping monitoring starts after ({warm_up_epochs} epochs)")
    print(f"  - Optimizer: AdamW, lr={learning_rate}, "
          f"epochs={Freeze_epoch}, batch={Batchsize}, seed=48")
    print("=" * 60)
