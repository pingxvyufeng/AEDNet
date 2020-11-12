#encoding:utf-8



import torch
import torch.nn as nn
import torch.nn.functional as F
from config import  *
from layer_norm import  *
import numpy as np

# self attention
class SelfAttention(nn.Module):

    def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.m = input_size
        self.output_size = output_size

        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.1)#设置dropout



    def forward(self, x):
        n = x.shape[0]  # sequence length

        K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size 参数X
        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size 参数Xi
        V = self.V(x)
        Q  = 0.06 * Q



        logits = torch.matmul(Q, K.transpose(1,0))#.transpose是调整数组的行列索引值，这个地方类似于转置

        if self.ignore_itself:
            # Zero the diagonal activations (a distance of each frame with itself)强制每帧和自身差异为0
            logits[torch.eye(n).byte()] = -float("Inf") #torch.eye(n)生成对角线全1，其余部分全0的二维数组 float("-inf")表示负无穷

        if self.apperture > 0:
            # Set attention to zero to frames further than +/- apperture from the current one当帧的距离过长将其注意力向量置为0
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)#torch.tril返回矩阵上三角部分，其余部分定义为0
            # -self.apperture为正数保留输入矩阵保留主对角线与主对角线以上除去apperture行的元素
            # torch.triu
            logits[trimask == 1] = -float("Inf")

        att_weights_ = nn.functional.softmax(logits, dim=-1)#dim=-1是对某一维度的行进行softmax运算
        weights = self.drop50(att_weights_)
        y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
        y = self.output_linear(y)

        return y, att_weights_



class MLP(nn.Module):
    def __init__(self,input_size,output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, output_size)
        #self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        m = 1024
        din = x.view(-1, m)
        #print (din.size())
        dout = nn.functional.tanh(self.fc1(din))

        dout = nn.functional.tanh(self.fc2(dout))


        #dout = nn.functional.tanh(self.fc3(dout))


        return dout


class AEDNet(nn.Module):

    def __init__(self):
        super(AEDNet, self).__init__()

        self.m = 1024 # cnn features size
        self.att = SelfAttention(input_size=self.m, output_size=self.m)

        self.mlp = MLP(input_size=1024,output_size=1024)
 
        self.ka = nn.Linear(in_features=self.m, out_features=1024)
        self.lstm = nn.GRU(input_size=1024, hidden_size=512, num_layers= 1, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(input_size=1024, hidden_size=1024, num_layers=1, batch_first=True, bidirectional=False)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)
        self.layer_norm_kb = LayerNorm(self.kb.out_features)
        self.layer_norm_kc = LayerNorm(self.kc.out_features)
        self.gru1_cell = nn.GRUCell(1024, 1024)
        self.gru1_drop = nn.Dropout(p=0.5)
        self.gru2_cell = nn.GRUCell(1024, 1024, bias=False)
        self.gru2_drop = nn.Dropout(p=0.5)

    def _init_lstm_state(self, d):
        bsz = d.size(0)
        return d.data.new(bsz, self.hidden_size).zero_(), d.data.new(bsz, self.hidden_size).zero_()



    def forward(self, x, seq_len):


        m = x.shape[2]# Feature size 1024

        x = x.view(-1, m)#view在pytorch中相当于重新分配 -1表示暂时缺省，按第二维度为m分配

        p = x.unsqueeze(0)#unsqueeze(arg)表示在第arg维增加一个维度值为1的维度)

        o,_= self.lstm(p)

        z = o.squeeze(0)#squeeze(arg)表示第arg维的维度值为1，则去掉该维度。否则tensor不变

        c, att_weights_ = self.att(z)


        y = c +  x



        #
        y = self.drop50(y)

        y = y.view(-1, m)  #

        y = y.unsqueeze(0)

        y, _ = self.gru(y)

        y = y.squeeze(0)
        y = self.layer_norm_y(y)

        y = self.mlp(y)

        y = self.drop50(y)

        y = self.relu(y)

        y = self.kd(y)

        y = self.sig(y)

        y = y.view(1, -1)

        return y, att_weights_


if __name__ == "__main__":
    pass


