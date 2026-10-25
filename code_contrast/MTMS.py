import torch.nn as nn
import torch


class MSMT_LE(nn.Module):
    def __init__(self,in_features,hidden_features,attention_layers,num_heads,gate_num,linear_drop,control_num,gate_True):
        super(MSMT_LE, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.proj = nn.Linear(in_features, int(in_features*6))
        self.hard_parameter_sharing = HardSharing(int(in_features*6), hidden_features, attention_layers, num_heads,
                                                  linear_drop,gate_num,control_num,gate_True)
        self.L_layer = L_Layer(int(in_features*6), hidden_features)
        self.H_layer = H_Layer(in_features*6,hidden_features)
        # self.G_layer = G_Layer(in_features*3, hidden_features)

    def forward(self, x):
        x_copy = x[:, 9:]
        x = self.proj(x[:, :9])
        features = self.hard_parameter_sharing(x,x_copy)
        results = []
        LE = self.L_layer(features)
        H = self.H_layer(features)
        #G = self.G_layer(features)
        results += [LE]
        #results += [G]
        results += [H]
        results = torch.stack(results, dim=1).squeeze(dim=2)
        return results
        # return LE

#================================ 辅助层 ==============================

class HardSharing(nn.Module):
    def __init__(self,in_features,hidden_features,attention_layers,num_heads,linear_drop,gate_num,control_num,gate_True):
        super(HardSharing, self).__init__()
        self.hidden_features = hidden_features
        self.num_heads = num_heads

        self.attentionblocks = nn.ModuleList(
            [AttentionLayer(in_features,hidden_features,num_heads,linear_drop,gate_num,control_num,gate_True)
             for i in range(attention_layers)])

    def forward(self,x,y):
        for block in self.attentionblocks:
            x = block(x,y)
        return x

class Expert(nn.Module):
    def __init__(self,in_features,gate_num,control_num):
        super(Expert, self).__init__()
        self.Multi_gate = nn.ModuleList(Gate(in_features) for i in range(gate_num))
        # self.Gatecontrol1 =  nn.Sequential(
        #     nn.Linear(control_num,control_num),
        #     nn.ReLU(),
        #     nn.Linear(control_num, gate_num),
        #     nn.ReLU(),
        #     nn.Linear(gate_num, gate_num)
        # )
        self.Gatecontrol1 = nn.Sequential(
            nn.Linear(control_num, gate_num),
            nn.ReLU(),
            nn.Linear(gate_num, gate_num)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(0.1)

    def forward(self,x,y):
        block_output = []
        for block in self.Multi_gate:
            block_output += [block(x)]
        block_output = torch.stack(block_output, dim=-1)
        y = self.Gatecontrol1(y)
        y = self.softmax(y)
        choose = self.drop(y)
        output = torch.einsum("bcg,bg->bc", [block_output, choose])
        return output



class Gate(nn.Module):
    def __init__(self,in_features):
        super(Gate, self).__init__()
        # self.linear1 = nn.Sequential(
        #     nn.Linear(in_features,int(in_features/2)),
        #     nn.ReLU(),
        #     nn.Linear(int(in_features/2),int(in_features/4)),
        #     nn.ReLU(),
        #     nn.Linear(int(in_features/4),int(in_features/2)),
        #     nn.ReLU(),
        #     nn.Linear(int(in_features/2),in_features),
        # )
        self.linear1 = nn.Sequential(
            nn.Linear(in_features, int(in_features / 4)),
            nn.ReLU(),
            nn.Linear(int(in_features / 4), in_features),
        )

    def forward(self,x):
        x = self.linear1(x)
        return x

class L_Layer(nn.Module):
    def __init__(self,in_features,hidden_features):
        super(L_Layer, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, int(hidden_features / 4)),
            nn.ReLU(),
            nn.Linear(int(hidden_features / 4), int(hidden_features / 8)),
            nn.ReLU(),
            nn.Linear(int(hidden_features / 8), int(hidden_features / 16)),
            nn.ReLU(),
            nn.Linear(int(hidden_features / 16), 1)
        )

    def forward(self, x):
        x = self.linear1(x)
        return x

class H_Layer(nn.Module):
    def __init__(self,in_features,hidden_features):
        super(H_Layer, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, int(hidden_features / 8)),
            nn.ReLU(),
            nn.Linear(int(hidden_features / 8), 1)
        )

    def forward(self, x):
        x = self.linear1(x)
        return x

class G_Layer(nn.Module):
    def __init__(self,in_features,hidden_features):
        super(G_Layer, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, int(hidden_features / 8)),
            nn.ReLU(),
            nn.Linear(int(hidden_features / 8), 1)
        )

    def forward(self, x):
        x = self.linear1(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self,in_features,hidden_features,num_heads,linear_drop,gate_num,control_num,gate_True):
        super(AttentionLayer,self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_heads = num_heads
        self.Gate = gate_True
        self.Expert = Expert(in_features,gate_num,control_num)

        self.qkv = nn.Linear(in_features,in_features*3,bias=True)
        self.attn_drop = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.dk = (in_features // num_heads) ** -0.5
        self.proj = nn.Linear(in_features, in_features)
        self.proj_drop = nn.Dropout(linear_drop)

        self.linear1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features)
        )
        self.drop = nn.Dropout(linear_drop)


    def forward(self,x,y):
        B,C = x.shape
        shortcut1 = x
        qkv = self.qkv(x)
        qkv = qkv.reshape(B,3,self.num_heads,C//self.num_heads)
        qkv = qkv.permute(1,0,2,3)
        q,k,v = qkv[0],qkv[1],qkv[2]

        attention = (q @ k.transpose(-2, -1))
        attention = attention * self.dk
        attention = self.softmax(attention)
        attention = self.attn_drop(attention)

        output = (attention @ v).transpose(1,2).reshape(B,C)
        output = self.proj(output)
        output = self.proj_drop(output)

        output = output + shortcut1

        shortcut2 = output
        output = self.linear1(output)
        output = self.drop(output)
        output = output + shortcut2
        shortcut3 = output

        if self.Gate == True:
            output = self.Expert(output,y)
            output = output+shortcut3

        return output
