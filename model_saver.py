"""
模型保存和加载工具模块
用于保存和加载FedGCDR的预训练知识
"""
import torch
import os
import json
from datetime import datetime


def save_checkpoint(server, clients, mlps, save_dir='checkpoints', filename=None, training_stage='fine_tuning'):
    """
    保存完整的模型检查点
    
    参数:
        server: Server对象列表
        clients: Client对象列表
        mlps: MLP模型列表
        save_dir: 保存目录
        filename: 保存文件名（可选，默认使用时间戳）
        training_stage: 当前训练阶段 ('knowledge', 'async', 'fine_tuning')
    
    返回:
        保存的文件路径
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if filename is None:
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'fedgcdr_checkpoint_{formatted_time}.pt'
    
    filepath = os.path.join(save_dir, filename)
    
    # 准备要保存的数据
    checkpoint = {
        # 服务器相关
        'servers': [],
        'mlps': [],
        'clients_knowledge': [],
        'metadata': {
            'num_domain': len(server),
            'num_users': len(clients),
            'timestamp': datetime.now().isoformat(),
            'embedding_size': server[0].args.embedding_size if server else None,
            'training_stage': training_stage,
            'target_domain': server[0].args.target_domain if server else None,
            'dataset': server[0].args.dataset if server else None,
            'num_domain': server[0].args.num_domain if server else None
        }
    }
    
    # 保存每个服务器的状态
    for idx, svr in enumerate(server):
        server_data = {
            'id': svr.id,
            'domain_name': svr.domain_name,
            'U': svr.U.cpu(),
            'V': svr.V.cpu(),
            'user_embedding_with_attention': svr.user_embedding_with_attention.cpu(),
            'item_embedding_with_attention': svr.item_embedding_with_attention.cpu() if hasattr(svr, 'item_embedding_with_attention') else torch.zeros_like(svr.V),
            'domain_attention': svr.domain_attention.cpu(),
            'item_gat_state_dict': svr.item_gat.state_dict(),
        }
        checkpoint['servers'].append(server_data)
    
    # 保存MLP模型参数
    for mlp in mlps:
        checkpoint['mlps'].append(mlp.state_dict())
    
    # 保存客户端知识向量（保持Tensor格式，避免序列化/反序列化的精度损失）
    for client in clients:
        # 将知识向量转换为列表格式存储
        knowledge_serialized = []
        for domain_knowledge in client.knowledge:
            if len(domain_knowledge) > 0:
                knowledge_serialized.append([k.tolist() if isinstance(k, torch.Tensor) else k 
                                            for k in domain_knowledge])
            else:
                knowledge_serialized.append([])
        checkpoint['clients_knowledge'].append(knowledge_serialized)
    
    # 保存到文件
    torch.save(checkpoint, filepath)
    print(f'模型检查点已保存到: {filepath}')
    print(f'训练阶段: {training_stage}')
    return filepath


def load_checkpoint(filepath, server, clients, mlps, device='cpu'):
    """
    从检查点加载模型参数
    
    参数:
        filepath: 检查点文件路径
        server: Server对象列表
        clients: Client对象列表
        mlps: MLP模型列表
        device: 设备
    
    返回:
        加载的元数据信息
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'检查点文件不存在: {filepath}')
    
    # 加载检查点
    checkpoint = torch.load(filepath, map_location=device)
    
    # 验证数据一致性
    num_domain = checkpoint['metadata']['num_domain']
    num_users = checkpoint['metadata']['num_users']
    training_stage = checkpoint['metadata'].get('training_stage', 'unknown')
    
    if len(server) != num_domain:
        print(f'警告: 服务器数量不匹配 (文件: {num_domain}, 当前: {len(server)})')
    if len(clients) != num_users:
        print(f'警告: 客户端数量不匹配 (文件: {num_users}, 当前: {len(clients)})')
    
    print(f'加载检查点，训练阶段: {training_stage}')
    
    # 加载每个服务器的状态
    for idx, server_data in enumerate(checkpoint['servers']):
        if idx < len(server):
            svr = server[idx]
            svr.U.data = server_data['U'].to(device)
            svr.V.data = server_data['V'].to(device)
            svr.user_embedding_with_attention.data = server_data['user_embedding_with_attention'].to(device)
            if 'item_embedding_with_attention' in server_data:
                svr.item_embedding_with_attention.data = server_data['item_embedding_with_attention'].to(device)
            svr.domain_attention.data = server_data['domain_attention'].to(device)
            svr.item_gat.load_state_dict(server_data['item_gat_state_dict'])
            print(f'已加载服务器 {idx} ({svr.domain_name}) 的参数')
    
    # 加载MLP模型参数
    for idx, mlp_state in enumerate(checkpoint['mlps']):
        if idx < len(mlps):
            mlps[idx].load_state_dict(mlp_state)
            print(f'已加载MLP模型 {idx} 的参数')
    
    # 加载客户端知识向量
    for idx, knowledge_serialized in enumerate(checkpoint['clients_knowledge']):
        if idx < len(clients):
            client = clients[idx]
            # 将序列化的知识转换回Tensor格式
            client.knowledge = []
            for domain_knowledge in knowledge_serialized:
                if len(domain_knowledge) > 0:
                    knowledge_tensors = [torch.tensor(k, device=device, dtype=torch.float32) if isinstance(k, list) else k 
                                         for k in domain_knowledge]
                    client.knowledge.append(knowledge_tensors)
                else:
                    client.knowledge.append([])
            print(f'已加载客户端 {idx} 的知识向量')
    
    print(f'模型检查点加载完成: {filepath}')
    print(f'元数据: {checkpoint["metadata"]}')
    
    return checkpoint['metadata']


def save_pretrained_embeddings(server, save_dir='embedding', filename=None):
    """
    仅保存预训练的嵌入向量（轻量级保存）
    
    参数:
        server: Server对象列表
        save_dir: 保存目录
        filename: 保存文件名
    
    返回:
        保存的文件路径
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if filename is None:
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'pretrained_embeddings_{formatted_time}.pt'
    
    filepath = os.path.join(save_dir, filename)
    
    embeddings = {}
    for svr in server:
        embeddings[svr.domain_name] = {
            'U': svr.U.cpu(),
            'V': svr.V.cpu(),
            'user_embedding_with_attention': svr.user_embedding_with_attention.cpu()
        }
    
    torch.save(embeddings, filepath)
    print(f'预训练嵌入向量已保存到: {filepath}')
    return filepath


def load_pretrained_embeddings(filepath, server, device='cpu'):
    """
    加载预训练的嵌入向量
    
    参数:
        filepath: 嵌入向量文件路径
        server: Server对象列表
        device: 设备
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'嵌入向量文件不存在: {filepath}')
    
    embeddings = torch.load(filepath, map_location=device)
    
    for svr in server:
        if svr.domain_name in embeddings:
            svr.U.data = embeddings[svr.domain_name]['U'].to(device)
            svr.V.data = embeddings[svr.domain_name]['V'].to(device)
            svr.user_embedding_with_attention.data = embeddings[svr.domain_name]['user_embedding_with_attention'].to(device)
            print(f'已加载 {svr.domain_name} 的嵌入向量')
    
    print(f'预训练嵌入向量加载完成: {filepath}')


def list_checkpoints(checkpoint_dir='checkpoints'):
    """
    列出所有可用的检查点文件
    
    参数:
        checkpoint_dir: 检查点目录
    
    返回:
        检查点文件信息列表
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pt'):
            filepath = os.path.join(checkpoint_dir, filename)
            stat = os.stat(filepath)
            checkpoints.append({
                'filename': filename,
                'filepath': filepath,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    
    # 按修改时间排序
    checkpoints.sort(key=lambda x: x['modified_time'], reverse=True)
    return checkpoints