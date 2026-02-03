import random
import numpy as np
import pandas as pd
import torch
import os
import json
import importlib

import math
import argparse
import warnings
import datetime
import utility
from model_saver import save_checkpoint, load_checkpoint, save_pretrained_embeddings, load_pretrained_embeddings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='args for fedgcdr')
parser.add_argument('--dataset', choices=['amazon', 'douban'], default='amazon')
parser.add_argument('--round_gat', type=int, default=2)
parser.add_argument('--round_ft', type=int, default=2)
parser.add_argument('--num_domain', type=int, default=4)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--target_domain', type=int, default=1)
parser.add_argument('--lr_mf', type=float, default=0.005)
parser.add_argument('--lr_gat', type=float, default=0.001)
parser.add_argument('--embedding_size', type=int, default=16)
parser.add_argument('--local_epoch', type=int, default=3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--num_negative', type=int, default=4)
parser.add_argument('--user_batch', type=int, default=16)
parser.add_argument('--model', type=str, default='fedgcdr')
parser.add_argument('--knowledge', type=bool, default=False)
parser.add_argument('--only_ft', type=bool, default=False)
parser.add_argument('--eps', type=float, default=8)
parser.add_argument('--dp', type=bool, default=True)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--num_users', type=int)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--description', type=str, default=None)
parser.add_argument('--save_checkpoint', type=bool, default=True, help='是否保存训练检查点')
parser.add_argument('--checkpoint_path', type=str, default=None, help='加载检查点的路径')
parser.add_argument('--save_pretrained', type=bool, default=True, help='是否保存预训练嵌入')
parser.add_argument('--pretrained_path', type=str, default=None, help='加载预训练嵌入的路径')
parser.add_argument('--skip_gat', type=bool, default=None, help='是否跳过GAT训练（加载检查点时自动设置为True）')
args = parser.parse_args()

Server = importlib.import_module('model.' + args.model + '.party').Server
Client = importlib.import_module('model.' + args.model + '.party').Client
MLP = importlib.import_module('model.' + args.model + '.model').MLP


device = torch.device(args.device)

domain_user, dic, domain_names = utility.set_dataset(args)
client_train_data, server_evaluate_data, num_items, num_users, user_dic = dic['client_train_data'], dic[
    'server_evaluate_data'], dic['num_items'], dic['num_users'], dic['user_dic']
clients = [Client(i, client_train_data[i], num_items, 0, domain_names, args) for i in range(args.num_users)]
server = [
    Server(i, domain_names[i], num_items[i], clients, domain_user[domain_names[i]], server_evaluate_data[i], user_dic,
           args) for i in range(args.num_domain)]
MLPs = [MLP(args.embedding_size).to(device) for _ in range(args.num_domain)]

# eval pre-train model
for it in server:
    it.test_mf(0)

# 加载检查点（如果指定了checkpoint_path）
if args.checkpoint_path:
    print(f'正在从检查点加载模型: {args.checkpoint_path}')
    metadata = load_checkpoint(args.checkpoint_path, server, clients, MLPs, device)
    print('检查点加载完成，跳过GAT训练阶段')
    args.skip_gat = True  # 自动跳过GAT训练
    args.only_ft = True  # 跳过ASYNC阶段，直接进入微调
else:
    print('未指定检查点，开始训练')
    metadata = None
    
tar_domain = args.target_domain
k_dic, emb_dic = {}, {}

now = datetime.datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S").replace(' ', '_').replace(':', "_")
formatted_date = now.strftime("%Y-%m-%d")
output_file = 'output/' + str(args.num_domain) + '_' + args.model + '_dp_' + str(args.dp) + '_tar_' + str(
    args.target_domain) + '_' + str(
    args.random_seed) + '_' + formatted_date_time + '.out'

with open(output_file, 'w') as f:
    f.write(str(args)+'\n')
print(args)

# load knowledge
if args.skip_gat:
    print('已加载检查点，跳过知识训练阶段')
    # 确保所有Client的mlp引用正确
    for client in clients:
        client.mlp = MLPs
elif args.knowledge:
    with open('knowledge_hr/' + str(args.num_domain) + 'domains.json', 'r') as f:
        k_dic = json.load(f)
    for i in range(args.num_users):
        clients[i].knowledge = k_dic[str(i)]
else:
    # 确保所有Client的mlp引用正确
    for client in clients:
        client.mlp = MLPs
    order = [i for i in range(args.num_domain)]
    for it in order:
        max_hr, max_ndcg, epoch_id, no_improve = 0, 0, 0, 0
        knowledge = [1] * args.num_users
        for i in range(args.round_gat):
            print(f'{server[it].domain_name} gat round {i}: ' + formatted_date_time)
            server[it].kt_stage()
            hr_5, ndcg_5, hr_10, ndcg_10 = server[it].test_gat(i)
            with open(output_file, 'a') as f:
                f.write(
                    f'[{server[it].domain_name} GAT Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
                    f' ndcg_10 = {ndcg_10:.4f}\n')
            print(
                f'[{server[it].domain_name} GAT Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
                f' ndcg_10 = {ndcg_10:.4f}\n')
            if hr_10 > max_hr or (hr_10 == max_hr and ndcg_10 > max_ndcg):
                no_improve = 0
                epoch_id = i
                max_hr = hr_10
                max_ndcg = ndcg_10
                for client in clients:
                    knowledge[client.id] = client.knowledge[it]
            else:
                no_improve += 1
            # if no_improve > 100:
            #     break
        for client in clients:
            client.knowledge[it] = knowledge[client.id]

    # save_knowledge
    for i in range(args.num_users):
        for kl in clients[i].knowledge:
            if len(kl) != 0:
                kl[0] = kl[0].tolist()
        k_dic[i] = clients[i].knowledge
    with open('knowledge_64/' + str(args.num_domain) + 'domains' + '_' + formatted_date + '.json', 'w') as f:
        json.dump(k_dic, f)
server[tar_domain].mlp = MLPs

# ASYNC(目标域知识激活)
if args.skip_gat:
    print('已加载检查点，跳过ASYNC阶段')
elif args.only_ft is False:
    max_hr, max_ndcg, epoch_id, no_improve = 0, 0, 0, 0
    for i in range(args.round_gat):
        print(f'{server[tar_domain].domain_name} gat round {i}: ' + formatted_date_time)

        try:
            server[tar_domain].kt_stage(True)
        except:
            continue
        hr_5, ndcg_5, hr_10, ndcg_10 = server[tar_domain].test_gat(i)
        with open(output_file, 'a') as f:
            f.write(
                f'[{server[tar_domain].domain_name} GAT Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
                f' ndcg_10 = {ndcg_10:.4f}\n')
        print(
            f'[{server[tar_domain].domain_name} GAT Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
            f' ndcg_10 = {ndcg_10:.4f}\n')
        if hr_10 > max_hr or (hr_10 == max_hr and ndcg_10 > max_ndcg):
            no_improve = 0
            epoch_id = i
            max_hr = hr_10
            max_ndcg = ndcg_10
            emb_dic[domain_names[tar_domain]] = [server[tar_domain].user_embedding_with_attention.data.tolist(),
                                                 server[tar_domain].V.data.tolist()]
        else:
            no_improve += 1
        # if no_improve > 100:
        #     break

    server[tar_domain].U = torch.tensor(emb_dic[domain_names[tar_domain]][0], device=args.device)
    server[tar_domain].V = torch.tensor(emb_dic[domain_names[tar_domain]][1], device=args.device)
    emb_dic['parser'] = vars(args)
    with open('embedding/' + args.model + '/' + str(args.num_domain) + 'dp' + str(args.dp) + '_' + args.dataset + '_' + domain_names[
        tar_domain] + '_' + args.model + '.json', 'w') as f:
        json.dump(emb_dic, f)

# load embedding
else:
    with open('embedding/' + args.model + '/' + str(args.num_domain) + 'dp' + str(args.dp) + '_' + args.dataset + '_' +
              domain_names[tar_domain] + '_' + args.model + '.json', 'r') as f:
        dic = json.load(f)
        tar_name = domain_names[args.target_domain]
        server[tar_domain].U.data, server[tar_domain].V.data = torch.tensor(dic[tar_name][0], device=args.device), \
            torch.tensor(dic[tar_name][1], device=args.device)

max_hr, max_ndcg, epoch_id, no_improve = 0, 0, 0, 0
max_hr_5, max_hr_10, max_ndcg_5, max_ndcg_10 = 0, 0, 0, 0
for i in range(args.round_ft):
    print(f'{server[tar_domain].domain_name} fine-tuning round {i} ' + formatted_date_time)
    server[tar_domain].mf_train()
    hr_5, ndcg_5, hr_10, ndcg_10 = server[tar_domain].test_mf(i)
    with open(output_file, 'a') as f:
        f.write(f'[{server[tar_domain].domain_name} Fine-tuning Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, '
                f'hr_10 ={hr_10:.4f}, ndcg_10 = {ndcg_10:.4f}\n')
    print(f'[{server[tar_domain].domain_name} Fine-tuning Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = '
          f'{hr_10:.4f}, ndcg_10 = {ndcg_10:.4f}\n')
    max_hr_5 = max(max_hr_5, hr_5)
    max_hr_10 = max(max_hr_10, hr_10)
    max_ndcg_5 = max(max_ndcg_5, ndcg_5)
    max_ndcg_10 = max(max_ndcg_10, ndcg_10)
    if hr_10 > max_hr or (hr_10 == max_hr and ndcg_10 > max_ndcg):
        no_improve = 0
        epoch_id = i
        max_hr = hr_10
        max_ndcg = ndcg_10
    else:
        no_improve += 1

with open(output_file, 'a') as f:
    f.write(str(epoch_id) + '\n')
    f.write(f'hr_5 = {max_hr_5}, ndcg_5 = {max_ndcg_5}, hr_10 = {max_hr_10}, ndcg_10 = {max_ndcg_10}')
print(epoch_id)
print(f'hr_5 = {max_hr_5}, ndcg_5 = {max_ndcg_5}, hr_10 = {max_hr_10}, ndcg_10 = {max_ndcg_10}')

# 保存检查点（如果启用）
if args.save_checkpoint:
    print('\n保存训练检查点...')
    checkpoint_filename = f'fedgcdr_checkpoint_{formatted_date_time}.pt'
    # 确定当前训练阶段
    if args.checkpoint_path:
        # 如果是从检查点加载的，则当前是微调阶段
        training_stage = 'fine_tuning'
    else:
        # 如果是全新训练，则根据是否完成GAT训练判断
        training_stage = 'fine_tuning'
    checkpoint_path = save_checkpoint(server, clients, MLPs, save_dir='checkpoints', 
                                    filename=checkpoint_filename, training_stage=training_stage)
    print(f'检查点已保存到: {checkpoint_path}')

# 保存预训练嵌入（如果启用）
if args.save_pretrained:
    print('\n保存预训练嵌入向量...')
    pretrained_filename = f'pretrained_embeddings_{formatted_date_time}.pt'
    pretrained_path = save_pretrained_embeddings(server, save_dir='embedding/pretrained', filename=pretrained_filename)
    print(f'预训练嵌入已保存到: {pretrained_path}')

print('\n训练完成！')
