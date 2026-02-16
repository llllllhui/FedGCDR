import torch.nn as nn
import torch

# LightGCN层实现
# LightGCN去除了特征变换和非线性激活，仅保留邻接聚合
# 专注于图结构信息的传播
class LightGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, adj):
        # x: 节点特征矩阵 (num_nodes, embedding_dim)
        # adj: 邻接矩阵 (num_nodes, num_nodes)
        # 简单的邻居聚合：h_i = sum_{j in N(i)} e_j / sqrt(|N(i)|*|N(j)|)
        
        # 归一化邻接矩阵（对称归一化）
        degree = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        
        # 对称归一化：D^(-1/2) * A * D^(-1/2)
        norm_adj = adj * d_inv_sqrt.view(-1, 1) * d_inv_sqrt.view(1, -1)
        
        # 邻居聚合
        output = torch.matmul(norm_adj, x)
        return output


class LightGCN(nn.Module):
    def __init__(self, args, in_feature, hid_feature=16, out_feature=16, num_layers=1, dropout=0.1):
        super().__init__()
        self.in_feature = in_feature
        self.hid_feature = hid_feature
        self.out_feature = out_feature
        self.num_layers = num_layers
        self.drop = nn.Dropout(p=dropout)
        
        # LightGCN使用多个层，每层都是简单的邻接聚合
        self.layers = nn.ModuleList([LightGCNLayer() for _ in range(num_layers)])
        
        # 残差连接投影层
        self.res_proj = nn.Linear(hid_feature, hid_feature) if num_layers > 1 else None

    @staticmethod
    def compute_ls(f_t, f_s):
        # 与GAT相同的知识相似度损失
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
        # 与GAT相同的MSE损失
        loss = 0
        for fs in f_s:
            loss += torch.nn.functional.mse_loss(fs, f_t)
        return loss

    def forward(self, x, is_transfer_stage=False, domain_attention=None, transfer_vec=None):
        """
        前向传播
        x: 输入特征矩阵，第一行是用户嵌入，其他是物品嵌入
        """
        ls, lm = 0, 0
        alpha, beta = 0.01, 0.01
        intermediate_embedding = []
        
        # 构建邻接矩阵（与原GAT相同）
        adj = torch.eye(len(x), device=x.device)
        adj[:, 0] = 1.
        adj[0, :] = 1.
        
        # LightGCN的核心：多层传播
        # 存储每一层的输出
        layer_outputs = []
        
        # 如果是转移阶段，先处理知识转移
        if is_transfer_stage:
            ls = alpha / 2 * self.compute_ls(x[0], transfer_vec)
            lm = beta / 2 * self.compute_lm(x[0], transfer_vec)
            transfer_vec = torch.stack(transfer_vec)
            x = torch.cat((x, transfer_vec))
            # 更新邻接矩阵
            adj = torch.eye(len(x), device=x.device)
            adj[:, 0] = 1.
            adj[0, :] = 1.
        else:
            # 非转移阶段：包含原始输入
            layer_outputs.append(x)
        
        # LightGCN的逐层传播，添加层归一化和残差连接
        for i, layer in enumerate(self.layers):
            x_new = layer(x, adj)
            # 动态创建LayerNorm在正确的设备上
            ln = nn.LayerNorm(self.hid_feature, device=x.device)
            x_new = ln(x_new)
            # 添加dropout
            x_new = self.drop(x_new)
            # 残差连接（如果有多个层）
            if self.res_proj is not None and i > 0:
                x_new = x_new + self.res_proj(x)
            x = x_new
            layer_outputs.append(x)
        
        # LightGCN使用所有层的平均作为最终表示
        # 非转移阶段：x_final = (1/(K+1)) * sum_{l=0}^{K} x^{(l)}
        # 转移阶段：x_final = (1/K) * sum_{l=0}^{K-1} x^{(l)}
        x_final = torch.stack(layer_outputs, dim=0).mean(dim=0)
        
        # 提取中间嵌入用于知识提取
        # 取第一层的用户嵌入
        intermediate_embedding.append(layer_outputs[0][0].data)
        
        return x_final, intermediate_embedding, ls, lm


# 核心功能: 知识向量转换（保持不变）
# MLP用于将源域知识转换为目标域知识
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