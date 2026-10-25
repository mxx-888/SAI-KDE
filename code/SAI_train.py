import torch
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from SAI import MSMT_LE
import math
import os
from sklearn.metrics import mean_squared_error



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
        input_data = data[:,i:i+1]
        max = np.max(input_data)
        min = np.min(input_data)
        maxlist.append(max)
        minlist.append(min)
        input_data = (input_data-min)/(max-min)
        data[:,i:i+1] = input_data
    return maxlist,minlist,data

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

train_ratio = 0.9
Batchsize = 128
learning_rate = 0.001
Init_epoch = 0
Freeze_epoch = 100

path = r'D:\SAI\SAI-main\experiment\exp1\train'
test_val = r'D:\SAI\SAI-main\experiment\exp1\validation'


max_min_data = get_data(r'D:\SAI\SAI-main\experiment\exp1\maxmin')


#datamaxmin = np.array(max_min_data[['WS','Ta','RH','Press','SW_IN','LAI','SIF','acc_rain7','SM','Month','DEM','GRA',
                                    #'WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float) # rain
datamaxmin = np.array(max_min_data[["acc_rain7","WS","Ta", "Press", "RH", "SW_IN",'SM','LAI','SIF',"Month",'DEM',
                 'GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float) # rain
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
# datarichx = np.array(datarich[["Rain","WS","Ta", "P", "RH", "SW_IN",'SM','LAI','SIF',"Month",'DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float) # rain
datavalx = np.array(dataval[['WS','Ta','RH','Press','SW_IN','LAI','SIF','acc_rain7','SM','Month','DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float) # rain
datavaly = np.array(dataval[["ET",'H']]).astype(float)
for i in range(datavalx.shape[1]):
    input_data = datavalx[:, i:i + 1]
    input_data = (input_data - minx[i]) / (maxx[i] - minx[i])
    datavalx[:, i:i + 1] = input_data

data = get_data(path)
# datatrainx = np.array(data[["Rain","WS","Ta", "P", "RH", "SW_IN",'SM','LAI','SIF',"Month",'DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float) # rain
datatrainx = np.array(data[['WS','Ta','RH','Press','SW_IN','LAI','SIF','acc_rain7','SM','Month','DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float) # rain
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
train_torch_dataset = Data.TensorDataset(train_x,train_y)

train_loader = DataLoader(train_torch_dataset,batch_size=Batchsize,shuffle=True,num_workers=0)

in_features = 9
sparameters = 7
num_heads = 3
hidden_features= 48
parameter_features = 20
gate_num = 6
control_num = 16
gate_True = True
attention_layers = 10
linear_drop = 0.1


class MyLoss(nn.Module):
    def __init__(self,a,b,c):
        super(MyLoss, self).__init__()
        self.a = a
        self.b = b
        self.c = c
    def forward(self,Pre,Test):
        ET = Test[:,0:1]
        ET_pre = Pre[:, 0:1]

        loss_ori = torch.mean(torch.pow(ET - ET_pre, 2))

        ET_low = torch.pow(ET,4)
        ET_low_pre = torch.pow(ET_pre,4)
        loss_add = torch.mean(torch.pow(ET_low - ET_low_pre, 2))

        H = Test[:,1:2]
        H_pre = Pre[:,1:2]
        loss_H = torch.mean(torch.pow(H - H_pre, 2))

        loss = self.a * loss_ori + (1-self.a - self.c) * loss_add + self.c * loss_H
        # loss = 0.7 * loss_ori + 0.3 * loss_H
        return loss


model = MSMT_LE(in_features,hidden_features,attention_layers,num_heads,gate_num,linear_drop,control_num,gate_True)
modeltrain = model.train()
# loss = MtLoss(0.4)
loss = MyLoss(0.8,0.1,0.1)
epoch_step = num_train // Batchsize


model_path      = r""
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

optimizer = torch.optim.SGD(modeltrain.parameters(), lr = learning_rate)
warm_up_cosine_lr = lambda epoch: epoch/warm_up_epochs if epoch<= warm_up_epochs else 0.5*(math.cos((epoch-warm_up_epochs)/(Freeze_epoch-warm_up_epochs)*math.pi)+1)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,warm_up_cosine_lr)

for epoch in range(Freeze_epoch):
    train_loss = []
    for step, (x_train, y_train) in enumerate(train_loader):
        output_train = modeltrain(x_train)
        lossvalue = loss(output_train, y_train)
        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()
        train_loss.append(lossvalue.item())
    lr_scheduler.step()
    train_loss = np.array(train_loss)
    train_loss = np.mean(train_loss)
    with torch.no_grad():
        validation_output = modeltrain(validationx_data)
    val = math.sqrt(mean_squared_error((np.array(validation_output)[:,0:1])*(maxy[0]-miny[0])+miny[0], validationy_data[:,0:1]))
    torch.save(modeltrain.state_dict(),r'D:\SAI\SAI-main\logs\SAI_' + str(epoch + 1) + 'train_' + str(train_loss) + 'val_' + str(val)  + '.pth')
