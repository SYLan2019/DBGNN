from __future__ import division

import argparse
import math
import time
import random
import os
import copy
import numpy as np
from torch import optim
import torch.nn.functional as F
from utils import load_metr_la_rdata, calculate_random_walk_matrix, load_pems04_data, load_pems08_data, \
    load_pemsbay_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
# 改自final_pems04_addval_res2_attentionfusion_loss.py 将输入经过2个3维图卷积后，残差和结果自注意力融合,对两个分支加入损失限制
#用于pems08数据集
def parse_arg():

    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='pems08',help='name of the datasets,select from metr,nrel,ushcn,sedata of pemsbay')
    parser.add_argument('--n_s',type=int,default=90,help='sampled space dimension')  # n_s=n_o+n_m
    parser.add_argument('--h',type=int,default=25,help='sampled time dimension')
    parser.add_argument('--z',type=int,default=100,help='hidden dimension for graph convolution')
    parser.add_argument('--K',type=int,default=1,help='if use diffusion convolution,the actual diffusion conv step is K+1')
    parser.add_argument('--n_m',type=int,default=40,help='number of mask node during training')
    parser.add_argument('--n_u',type=int,default=80,help='target locations,n_u locations will be deleted from training data')
    parser.add_argument('--epochs',type=int,default=500,help='max training episode')
    parser.add_argument('--learning_rate',type=float,default=0.0005,help='the learning rate')
    parser.add_argument('--E_maxvalue',type=int,default=50,help='the max value from experience')
    parser.add_argument('--batch_size',type=int,default=4,help='batch_size')
    parser.add_argument('--to_plot',type=bool,default=True,help='Whether to plot the RMSE training result')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--id', type=int, default=1, help='id')
    parser.add_argument('--gpu_id',type=int,default=0,help='which gpu 0 or 1')
    parser.add_argument('--blocks',type=int,default=4)      #可以改
    parser.add_argument('--layers',type=int,default=2)      #可以改
    parser.add_argument('--dropout',type=int,default=0.1)   #可以改
    parser.add_argument('--residual',type=int,default=32,help='the channels of residual')   #可以
    parser.add_argument('--dilation', type=int, default=32, help='the channels of dilation')    #可以
    parser.add_argument('--skip',type=int,default=256)  #可以
    parser.add_argument('--end',type=int,default=512)   #可以
    parser.add_argument('--patience',type=int,default=30)
    args=parser.parse_args()
    return args

class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """
    def __init__(self, in_channels, out_channels, orders=1, activation = 'relu'):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1         #这是什么东西？
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                             out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)          #用U（-stdv,stdv)的均匀分布填充theta1,作为重置
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)          #重置偏置bias

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size = X.shape[0] # batch_size
        num_node = X.shape[1]
        input_size = X.size(2)  # time_length       #和in_channels一个东西
        supports = []
        supports.append(A_q)
        supports.append(A_h)

        x0 = X.permute(1, 2, 0) #(num_nodes, num_times, batch_size)   交换维度
        x0 = torch.reshape(x0, shape=[num_node, input_size * batch_size])   #input_size也即num_timesteps
        x = torch.unsqueeze(x0, 0)          # x.shape=[1，num_node,input_size*batch_size]
        for support in supports:            # support.shape=[num_nodes,num_nodes]
            x1 = torch.mm(support, x0)      #x1.shape=[num_nodes,input_size*batch_size]
            x = self._concat(x, x1)         #在第0维拼接，每循环一次x.shape[0]++ x.shape=[2,num_nodes,input_size*batch_size],注意_concat会给第二个补维度
            for k in range(2, self.orders + 1):         #x.shape=[2*orders+1,num_nodes,input_size*batch_size],注意_concat会给第二个补维度
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, num_node, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)    Theta1.shape=[input_size*matrics,output]
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)

        return x

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()
    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A)) # b,2,207,12 * 207,207---->b,2,207,12
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
    def forward(self,x):
        return self.mlp(x)
class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):    # Support_len is the corresponding number of adjacency matrices,
                                                                    # to be modified according to actual conditions
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order
    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class DGL(nn.Module):   #期望输入bfnt，输出nn
    def __init__(self,time_dim,feature):
        super(DGL,self).__init__()
        self.d_k=32                         # 可以改
        self.dim=time_dim*feature
        self.W_Q=nn.Linear(self.dim,self.d_k,bias=False)
        self.W_K=nn.Linear(self.dim,self.d_k,bias=False)
    def drop(self,tensor):
        k = int(tensor.shape[0]*tensor.shape[1]*0.1)  # 要保留的最大值数量
        dropout_prob = 0.1

        # 找到张量中的前k个最大值及其位置
        values, indices = torch.topk(tensor.view(-1), k)
        topk_mask = torch.zeros_like(tensor)
        topk_mask.view(-1)[indices] = 1
        topk_mask = topk_mask.view(tensor.size())
        # 复制张量并保留前k个最大值
        tensor_copy = tensor.clone()
        tensor_copy = tensor_copy * topk_mask
        # 对剩余部分应用dropout
        dropout_mask = torch.rand_like(tensor) < dropout_prob
        tensor_remaining = tensor * (1 - topk_mask)  # 剩余部分
        # tensor_remaining = tensor_remaining * dropout_mask
        tensor_remaining=F.dropout(tensor_remaining,p=dropout_prob,training=self.training)
        # 合并保留的最大值和dropout后的剩余部分
        final_tensor = tensor_copy + tensor_remaining
        return final_tensor

    def forward(self,x):    # bfnt
        residual, batch_size, feature, num_nodes = x, x.size(0), x.size(1), x.size(2)


        x=x.permute(0,2,3,1)    #bntf
        x=torch.reshape(x,shape=(batch_size,num_nodes,-1))

        Q=self.W_Q(x)
        K=self.W_K(x)
        scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(self.d_k)

        attn=nn.Softmax(dim=-1)(torch.sum(scores,dim=0))
        # attn = F.dropout(attn, 0.1, training=self.training)
        attn=self.drop(attn)
        return attn

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qfc=nn.Linear(2,64,bias=True)
        self.kfc=nn.Linear(2,64,bias=True)
        self.vfc=nn.Linear(2,64,bias=True)
        self.d=64
        self.li=nn.Linear(64,1)
    def forward(self,x,):   #btn2
        q=self.qfc(x)
        k=self.kfc(x)
        v=self.vfc(x)
        att=torch.matmul(q,k.transpose(-1,-2))/np.sqrt(self.d)
        att=torch.softmax(att,dim=-1)
        res=torch.matmul(att,v)     #btn64
        res=self.li(res).squeeze(-1)
        return res

class gwnet(nn.Module):
    def __init__(self, dropout=0.1,  in_dim=295,out_dim=25,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=3,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dgl=nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_gcn1=D_GCN(in_channels=out_dim,out_channels=100)
        self.start_gcn2 = D_GCN(in_channels=100, out_channels=out_dim)
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels,  kernel_size=(1,1))
        input_len=out_dim
        receptive_field = 1
        self.supports_len = 2
        self.total_time_len=0
        self.se=nn.ModuleList()

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                self.se.append(SEnet(dilation_channels))
                self.gconv.append(gcn(dilation_channels,residual_channels,dropout))
        for b in range(blocks):
            additional_scope=kernel_size-1
            for i in range(layers):
                input_len-=additional_scope
                additional_scope*=2
                self.dgl.append(DGL(time_dim=input_len,feature=dilation_channels))
                self.total_time_len=self.total_time_len+input_len
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,out_channels=out_dim,kernel_size=(1,1), bias=True)
        self.end_conv_t=nn.Linear(self.total_time_len,1)
        self.receptive_field = receptive_field
        self.a=Attention()

    def onehot(self, x):    #btn3-->btn295
        tod = F.one_hot(x[:, :, :, 1].long(), num_classes=288)
        dow = F.one_hot(x[:, :, :, 2].long(), num_classes=7)
        x = torch.cat((tod, dow), dim=-1) * x[:, :, :, :1]
        return x

    def forward(self, input,A_q,A_h):   # input.shape=[b,t,n,3]
        x_aux=self.start_gcn1(input[:,:,:,0].permute(0,2,1),A_q,A_h)
        x_aux=self.start_gcn2(x_aux,A_q,A_h).permute(0,2,1)
        input=self.onehot(input)    #btn295
        input=input.permute(0,3,2,1)    #b295nt
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)  #b295nt-->b32nt
        skip = []
        Adlist=[]
        for i in range(self.blocks * self.layers):
            residual = x    #b32nt

            #G-TCN
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x=self.se[i](x)

            #Skip-connection
            s = x
            s = self.skip_convs[i](s)
            skip.append(s)

            #DGL
            supports = []
            supports.append(A_q)
            supports.append(A_h)
            A_d=self.dgl[i](x)
            Adlist.append(A_d)
            supports.append(A_d)

            #DGCN & resconnection
            x = self.gconv[i](x,supports)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        skip=torch.cat(skip,dim=-1)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x=self.end_conv_t(x)
        x_main=torch.squeeze(x,-1)
        x=torch.stack((x_main,x_aux),dim=-1)
        x=self.a(x)
        return x,x_aux,x_main

class SEnet(nn.Module):
    def __init__(self,channels,ratio=4):
        super(SEnet, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio,False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_=x.size()
        avg=self.avgpool(x).view(b,c)
        fc=self.fc(avg).view(b,c,1,1)
        return fc*x
#***************************************************************************************************************************************
def load_data(dataset):
    '''Load dataset
    Input: dataset name
    Returns
    -------
    A: adjacency matrix
    X: processed data
    capacity: only works for NREL, each station's capacity
    '''

    tod=True
    dow=True
    if dataset == 'metr':
        A, X = load_metr_la_rdata()
        X = X[:, 0:1, :]    #n 1 t
        X=X.transpose(2,0,1)
    elif dataset=='pems04':
        X,A=load_pems04_data()
        X=X[:,:,2:] #tnf
    elif dataset=='pems08':
        X,A=load_pems08_data()
        X=X[:,:,2:] #tnf
    elif dataset=='pemsbay':
        X,A=load_pemsbay_data() #nt
        X=np.expand_dims(X,axis=0)  #fnt
        X=X.transpose(2,1,0)    #tnf

    #tnf
    if tod:
        MAX_TOD = 288   #288
        tod = [(i % MAX_TOD) for i in range(X.shape[0])]

        tod = [t / MAX_TOD for t in tod]
        tod = np.tile(tod, [1, X.shape[1], 1]).transpose((2, 1, 0))
        data = np.concatenate((X, tod), axis=-1)
    if dow:
        MAX_TOD = 288   #288
        MAX_DOW = 7
        dow = [((i // MAX_TOD) % MAX_DOW) for i in range(data.shape[0])]
        dow = np.tile(dow, [1, data.shape[1], 1]).transpose((2, 1, 0))
        data = np.concatenate((data, dow), axis=-1)
    X=data.transpose(1,0,2) #ntf
    split_line1 = int(X.shape[1] * 0.7)

    training_set = X[:, :split_line1,:].transpose(1,0,2)
    print('training_set', training_set.shape)
    test_set = X[:, split_line1:,:].transpose(1,0,2)  # split the training and test period

    test_node=np.load('data/pems08/testnode.npz')
    unknow_set=test_node['arr_{}'.format(args.seed-1)]
    print("test_node:")
    print(unknow_set)
    unknow_set = set(unknow_set)

    full_set = set(range(0, X.shape[0]))
    know_set = full_set - unknow_set
    training_set_s = training_set[:, list(know_set),:]  # get the training data in the sample time period
    A_s = A[:, list(know_set)][list(know_set), :]  # get the observed adjacent matrix from the full adjacent matrix,
    # the adjacent matrix are based on pairwise distance,
    return A, X, training_set, test_set, unknow_set, full_set, know_set, training_set_s, A_s


"""
Define the test error
"""


def test_error(STmodel, unknow_set, test_data, A_s, Missing0, device):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """
    test_truth=test_data[:,:,0]  #tn
    unknow_set = set(unknow_set)
    time_dim=25
    test_omask = np.ones(test_data.shape)
    test_truth_omask=np.ones(test_truth.shape)  #tn
    if Missing0 == True:
        test_omask[test_data == 0] = 0
        test_truth_omask[test_truth == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')    #tn3
    test_truth=(test_truth*test_truth_omask).astype('float32')
    test_inputs_s = test_inputs

    missing_index = np.ones(np.shape(test_data)) #We found that there are irregular 0 values for METR-LA, so we treat those 0 values as missing data,
    missing_index[:, list(unknow_set),:] = 0    #Missing points are represented by 0
    missing_index_s = missing_index
    missing_index_truth=missing_index[:,:,0]

    o = np.zeros([test_data.shape[0] // time_dim * time_dim,test_inputs_s.shape[1]])
    x_aux= np.zeros([test_data.shape[0] // time_dim * time_dim,test_inputs_s.shape[1]])
    x_main= np.zeros([test_data.shape[0] // time_dim * time_dim,test_inputs_s.shape[1]])
    for i in range(0, test_data.shape[0] // time_dim * time_dim, time_dim):
        inputs = test_inputs_s[i:i + time_dim, :]
        missing_inputs = missing_index_s[i:i + time_dim, :,:]
        T_inputs = inputs * missing_inputs
        T_inputs = T_inputs / E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis=0)
        T_inputs = torch.from_numpy(T_inputs.astype('float32')).to(device)  #bnt

        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32')).to(device)
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32')).to(device)

        imputation,x_a,x_m = STmodel(T_inputs, A_q, A_h)
        imputation = imputation.cuda().data.cpu().numpy()
        x_a = x_a.cuda().data.cpu().numpy()
        x_m = x_m.cuda().data.cpu().numpy()
        o[i:i + time_dim, :] = imputation[0, :, :]
        x_aux[i:i + time_dim, :] = x_a[0, :, :]
        x_main[i:i + time_dim, :] = x_m[0, :, :]

    o = o * E_maxvalue
    x_aux=x_aux*E_maxvalue
    x_main=x_main*E_maxvalue
    truth = test_truth[0:test_set.shape[0] // time_dim * time_dim]
    o[missing_index_truth[0:test_set.shape[0] // time_dim * time_dim] == 1] = truth[
        missing_index_truth[0:test_set.shape[0] // time_dim * time_dim] == 1]

    real_o=np.copy(o)
    test_mask = 1 - missing_index_truth[0:test_set.shape[0] // time_dim * time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
    MAE = np.sum(np.abs(o - truth)) / np.sum(test_mask)
    RMSE = np.sqrt(np.sum((o - truth) * (o - truth)) / np.sum(test_mask))
    MAPE = np.sum(np.abs(o - truth) / (truth + 1e-5)) / np.sum(test_mask)
    return MAE, RMSE, MAPE, real_o, truth,x_aux,x_main
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

if __name__ == "__main__":

    args = parse_arg()
    dataset = args.dataset
    n_o_n_m = args.n_s
    h = args.h
    z = args.z
    K = args.K
    n_m = args.n_m
    n_u = args.n_u
    max_iter = args.epochs
    learning_rate = args.learning_rate
    E_maxvalue = args.E_maxvalue
    batch_size = args.batch_size
    device = torch.device(f"cuda:{args.gpu_id}")
    seed = args.seed
    exp_id=args.id
    patience=args.patience
    logging.basicConfig(filename=f'log/{dataset}/{exp_id}.log',level=logging.INFO)
    logging.info('[experiment_id={}]  seed={}  z={}  K={}  learning_rate={}  E={}  batch={} epoch={}'.format(exp_id,seed,z,K,learning_rate,E_maxvalue,batch_size,max_iter))
    logging.info(f'n_s={n_o_n_m}, n_m={n_m}, n_u={n_u}')
    logging.info(f'residual={args.residual},dilation={args.dilation},skip={args.skip},end={args.end},blocks={args.blocks},layers={args.layers}')
    seed_torch(seed)
    save_path = "./result_best/%s/%s" % (dataset,exp_id)

    # load dataset
    A, X, training_set, test_set, unknow_set, full_set, know_set, training_set_s, A_s= load_data(dataset)
    # Define model
    STmodel = gwnet(residual_channels=args.residual,dilation_channels=args.dilation,skip_channels=args.skip,end_channels=args.end,blocks=args.blocks,layers=args.layers) # The graph neural networks
    STmodel.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(STmodel.parameters(), lr=learning_rate)
    RMSE_list = []
    MAE_list = []
    MAPE_list = []
    pred = []
    truth = []
    print('##################################    start training    ##################################')
    print(f"use gpu {args.gpu_id},  seed={seed},  experiment_id={exp_id}")
    best_mae = 100000
    early_stop_trigger=0
    for epoch in range(max_iter):
        time_s = time.time()
        for i in range(training_set.shape[0] // (h * batch_size)):
            t_random = np.random.randint(0, high=(training_set_s.shape[0] - h), size=batch_size, dtype='l')
            know_mask = set(random.sample(range(0, training_set_s.shape[1]), n_o_n_m))  # sample n_s nodes

            feed_batch = []
            for j in range(batch_size):
                feed_batch.append(training_set_s[t_random[j]:t_random[j] + h, list(know_mask)])
            inputs = np.array(feed_batch)

            inputs_omask = np.ones(np.shape(inputs))    #b t n 3
            inputs_omask[inputs == 0] = 0  # We found that there are irregular 0 values for METR-LA, so we treat those 0 values as missing data,
            inputs_omask[:,:,:,1:3]=1
            missing_index = np.ones((inputs.shape)) #btn3
            for j in range(batch_size):
                missing_mask = random.sample(range(0, n_o_n_m), n_m)  # Masked locations
                missing_index[j, :, missing_mask] = 0

            Mf_inputs = inputs * inputs_omask * missing_index / E_maxvalue  # normalize the value according to experience
            Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32')).to(device)
            mask = torch.from_numpy(inputs_omask.astype('float32')).to(device)  # The reconstruction errors on irregular 0s are not used for training

            A_dynamic = A_s[list(know_mask), :][:, list(know_mask)]  # Obtain the adjacent matrix
            A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32')).to(device)
            A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32')).to(device)

            truth = torch.from_numpy(inputs / E_maxvalue).to(device)  # The label

            optimizer.zero_grad()
            X_res,X_aux,X_main = STmodel(Mf_inputs, A_q, A_h)  # Obtain the reconstruction
            mask=mask[:,:,:,0]
            truth=truth[:,:,:,0].float()
            loss = criterion(X_res * mask, truth * mask)+0.5*criterion(X_aux * mask, truth * mask)+0.5*criterion(X_main * mask, truth * mask)
            loss.backward()
            optimizer.step()  # Errors backward

        MAE_t, RMSE_t, MAPE_t, pred, truth,x_aux,x_main = test_error(STmodel, unknow_set, test_set, A, True, device)
        time_e = time.time()
        RMSE_list.append(RMSE_t)
        MAE_list.append(MAE_t)
        MAPE_list.append(MAPE_t)
        print(epoch, MAE_t, RMSE_t, MAPE_t, 'time=', time_e - time_s)

        if MAE_t < best_mae:
            best_mae = MAE_t
            best_rmse = RMSE_t
            best_mape = MAPE_t
            best_epoch = epoch
            best_model = copy.deepcopy(STmodel.state_dict())
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.savez(save_path + "result.npz", pred=pred, truth=truth,unknow_set=list(unknow_set))
            early_stop_trigger=0
        else:
            early_stop_trigger+=1
            if early_stop_trigger >= patience:
                print('early stop at epoch %d ' % (epoch))
                break

    torch.save(best_model, f'model/{dataset}_{exp_id}.pth')  # Save the model
    print("###############     best_result:        ")
    logging.info(f'best epoch={best_epoch}, mae={best_mae}, rmse={best_rmse}, mape={best_mape}')
    print("epoch = ", best_epoch, "     mae = ", best_mae, "     rmse = ", best_rmse, "     mape = ", best_mape)





