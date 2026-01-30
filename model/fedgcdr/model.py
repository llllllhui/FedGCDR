import torch.nn as nn
import torch

# GATLayer实现了图注意力网络(Graph Attention Network)的基本单元,
# 用于学习图中节点之间的关系权重。

# 单个图注意力层，使用可学习的注意力向量 A 计算节点间的注意力系数，并应用 softmax 激活函数
# 来归一化注意力权重。前向传播方法接收输入特征和邻接矩阵，通过线性变换的组合计算注意力输出
class GATLayer(nn.Module):
    def __init__(self, in_feature, out_feature, alpha):
        super().__init__()
        #__`in_feature`__: 输入特征维度 (例如16)
        self.in_feature = in_feature
        #__`out_feature`__: 输出特征维度 (例如16)
        self.out_feature = out_feature
        # 注意力矩阵A,`A`的形状: `(2 * out_feature, 1)`
        self.A = nn.Parameter(torch.empty(size=(2 * out_feature, 1)))
        nn.init.xavier_uniform_(self.A.data, nn.init.calculate_gain('relu'))
        self.alpha = alpha

    #向前传播
    def forward(self, input, adj):
        h = input
        h1 = torch.matmul(h, self.A[self.out_feature:, :])
        h2 = torch.matmul(h, self.A[:self.out_feature, :])
        e = h1 + h2.T
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=-1)
        ah = torch.matmul(attention, h)
        return ah


class GAT(nn.Module):
    def __init__(self, args, in_feature, hid_feature=16, out_feature=16, alpha=0.1, dropout=0):
        super().__init__()
        self.in_feature = in_feature
        self.hid_feature = hid_feature
        self.out_feature = out_feature
        self.drop = nn.Dropout(p=dropout)
        self.in2hidden = GATLayer(in_feature, hid_feature, alpha).to(args.device)
        self.hidden2out = GATLayer(hid_feature, out_feature, alpha).to(args.device)

    @staticmethod
    def compute_ls(f_t, f_s):
        total_sim = 0
        for fs in f_s:
            with torch.no_grad():
                sim = (torch.cosine_similarity(fs, f_t, dim=0) + 1) / 2
                total_sim += sim
        F_s = 0
        for fs in f_s:
            with torch.no_grad():
                sim = (torch.cosine_similarity(fs, f_t, dim=0) + 1) / 2
            F_s += sim * fs / total_sim
        loss = torch.norm(f_t - F_s) ** 2
        return loss

    @staticmethod
    def compute_lm(f_t, f_s):
        loss = 0
        for fs in f_s:
            loss += torch.nn.functional.mse_loss(fs, f_t)
        return loss

    def forward(self, x, is_transfer_stage=False, domain_attention=None, transfer_vec=None):
        ls, lm = 0, 0
        alpha, beta = 0.01, 0.01
        intermediate_embedding = []
        adj = torch.eye(len(x), device=x.device)
        adj[:, 0] = 1.
        adj[0, :] = 1.
        x = self.in2hidden(x, adj)
        intermediate_embedding.append(x[0].data)
        if is_transfer_stage:
            ls = alpha / 2 * self.compute_ls(x[0], transfer_vec)
            lm = beta / 2 * self.compute_lm(x[0], transfer_vec)
            transfer_vec = torch.stack(transfer_vec)
            x = torch.cat((x, transfer_vec))
            adj = torch.eye(len(x), device=x.device)
            adj[:, 0] = 1.
            adj[0, :] = 1.
        x = self.hidden2out(x, adj)
        return x, intermediate_embedding, ls, lm

#核心功能: 知识向量转换
#  __工作流程__
# 1. __输入__: 从其他领域学到的知识向量 (维度16)
# 2. __转换__: 通过MLP进行非线性变换
# 3. __输出__: 适配目标域的知识向量 (维度仍为16)
class MLP(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.L1 = nn.Linear(in_feature, 2*in_feature)
        self.L2 = nn.Linear(2*in_feature, int(in_feature/2))
        self.L3 = nn.Linear(int(in_feature/2), in_feature)
        self.f = nn.Tanh()

    def forward(self, x):
        x = self.f(self.L1(x))
        x = self.f(self.L2(x))
        x = self.f(self.L3(x))
        return x
