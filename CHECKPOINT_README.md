# Checkpoint 使用说明

## 功能概述

Checkpoint机制允许你在训练的关键阶段保存模型状态，并在后续训练中直接从这些阶段恢复，从而跳过已完成的训练步骤。

### 支持的Checkpoint阶段

1. **知识获取阶段 (KG)**: 所有域的GAT/LightGCN训练完成后
2. **知识转移阶段 (KT)**: 目标域的知识转移训练完成后
3. **微调阶段 (FT)**: 完整训练结束后（可选）

## 使用方法

### 1. 正常训练（自动保存Checkpoint）

默认情况下，训练会自动在关键阶段保存checkpoint：

```bash
python main.py --dataset amazon --num_domain 4 --target_domain 1 --gnn_type lightgcn
```

训练过程中会自动：
- 在知识获取阶段结束后保存 `kg_amazon_4domains_YYYYMMDD_HHMMSS`
- 在知识转移阶段结束后保存 `kt_amazon_4domains_target1_YYYYMMDD_HHMMSS`

### 2. 列出可用Checkpoint

```bash
python main.py --list_checkpoints
```

输出示例：
```
================================================================================
可用的Checkpoints (目录: checkpoints)
================================================================================

【知识获取阶段】
  1. kg_amazon_4domains_20250223_143020
     路径: checkpoints/kg_amazon_4domains_20250223_143020
     时间: 2025-02-23T14:30:20
     大小: 45.23 MB

【知识转移阶段】
  1. kt_amazon_4domains_target1_20250223_150530
     路径: checkpoints/kt_amazon_4domains_target1_20250223_150530
     时间: 2025-02-23T14:05:30
     大小: 52.67 MB
```

### 3. 从知识获取阶段恢复训练

跳过知识获取阶段，直接从知识转移阶段开始训练：

```bash
python main.py \
    --dataset amazon \
    --num_domain 4 \
    --target_domain 1 \
    --gnn_type lightgcn \
    --resume_from kg \
    --checkpoint_path checkpoints/kg_amazon_4domains_20250223_143020
```

这将：
- ✓ 加载所有域的模型状态
- ✓ 加载所有客户端的知识向量
- ✓ 跳过知识获取阶段（约30轮 × 4域的训练）
- ✓ 直接进入目标域的知识转移阶段

### 4. 从知识转移阶段恢复训练

跳过知识获取和转移阶段，直接从微调阶段开始训练：

```bash
python main.py \
    --dataset amazon \
    --num_domain 4 \
    --target_domain 1 \
    --gnn_type lightgcn \
    --resume_from kt \
    --checkpoint_path checkpoints/kt_amazon_4domains_target1_20250223_150530
```

这将：
- ✓ 加载所有域的模型状态
- ✓ 加载所有客户端的知识向量
- ✓ 加载目标域的额外状态（注意力机制等）
- ✓ 加载MLP模型
- ✓ 跳过知识获取和转移阶段
- ✓ 直接进入微调阶段

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--save_checkpoint` | flag | True | 是否保存checkpoint |
| `--resume_from` | str | None | 恢复阶段: `kg` 或 `kt` |
| `--checkpoint_path` | str | None | checkpoint目录路径 |
| `--list_checkpoints` | flag | False | 列出所有可用checkpoint |
| `--checkpoint_dir` | str | checkpoints | checkpoint保存目录 |

## Checkpoint目录结构

```
checkpoints/
├── kg_amazon_4domains_20250223_143020/        # 知识获取阶段
│   ├── metadata.json                           # 元数据（参数、时间戳等）
│   ├── models.pt                               # 模型状态
│   ├── knowledge.pt                            # 客户端知识
│   └── summary.json                            # 摘要信息
│
└── kt_amazon_4domains_target1_20250223_150530/ # 知识转移阶段
    ├── metadata.json                           # 元数据
    ├── models.pt                               # 模型状态
    ├── knowledge.pt                            # 客户端知识
    ├── target_state.pt                         # 目标域额外状态
    ├── mlp.pt                                  # MLP模型
    └── summary.json                            # 摘要信息
```

## 参数兼容性验证

加载checkpoint时会自动验证以下关键参数：

- `dataset`: 数据集名称
- `num_domain`: 域数量
- `embedding_size`: 嵌入维度
- `round_gat`: GAT训练轮数
- `round_ft`: 微调轮数

如果参数不匹配，会显示详细错误信息并退出。

## Checkpoint管理

### 自动清理

默认保留最近3个checkpoint，自动删除较旧的checkpoint以节省磁盘空间。

可通过修改 `checkpoint.py` 中的 `max_keep` 参数调整：

```python
checkpoint_manager = CheckpointManager(checkpoint_dir='checkpoints', max_keep=5)
```

### 手动清理

```bash
# 删除特定checkpoint
rm -rf checkpoints/kg_amazon_4domains_20250223_143020

# 删除所有checkpoint
rm -rf checkpoints/*
```

## 使用场景

### 场景1: 调试知识转移阶段

发现问题在知识转移阶段，需要多次调试：

```bash
# 第一次运行：训练并保存checkpoint
python main.py --dataset amazon --num_domain 4 --target_domain 1

# 后续调试：直接从知识转移阶段开始
python main.py --dataset amazon --num_domain 4 --target_domain 1 \
    --resume_from kg \
    --checkpoint_path checkpoints/kg_amazon_4domains_20250223_143020
```

### 场景2: 对比不同微调策略

保持知识获取和转移阶段不变，尝试不同的微调参数：

```bash
# 基础训练
python main.py --dataset amazon --num_domain 4 --target_domain 1 --round_ft 60

# 实验1: 更长的微调
python main.py --dataset amazon --num_domain 4 --target_domain 1 --round_ft 100 \
    --resume_from kt \
    --checkpoint_path checkpoints/kt_amazon_4domains_target1_20250223_150530

# 实验2: 更小的学习率
python main.py --dataset amazon --num_domain 4 --target_domain 1 --lr_mf 0.001 \
    --resume_from kt \
    --checkpoint_path checkpoints/kt_amazon_4domains_target1_20250223_150530
```

### 场景3: 不同目标域的实验

使用相同的知识获取结果，训练不同的目标域：

```bash
# 基础训练（目标域=0）
python main.py --dataset amazon --num_domain 4 --target_domain 0

# 使用相同的KG结果，训练目标域=1
python main.py --dataset amazon --num_domain 4 --target_domain 1 \
    --resume_from kg \
    --checkpoint_path checkpoints/kg_amazon_4domains_20250223_143020

# 使用相同的KG结果，训练目标域=2
python main.py --dataset amazon --num_domain 4 --target_domain 2 \
    --resume_from kg \
    --checkpoint_path checkpoints/kg_amazon_4domains_20250223_143020
```

## 注意事项

1. **参数一致性**: 恢复训练时，除了 `target_domain` 和微调相关参数外，其他参数应与保存时一致

2. **设备兼容性**: Checkpoint可以在CPU和GPU之间迁移，会自动处理设备转换

3. **磁盘空间**: 每个checkpoint大约占用40-60MB，建议定期清理旧checkpoint

4. **版本兼容性**: 确保保存和加载时使用相同版本的代码

5. **文件完整性**: 不要手动修改checkpoint目录下的文件，可能导致加载失败

## 故障排除

### 问题1: "参数不匹配"错误

**原因**: 当前参数与checkpoint保存时的参数不一致

**解决**: 检查并修正参数，特别是 `dataset`, `num_domain`, `embedding_size` 等

### 问题2: "未找到域 X 的状态"

**原因**: Checkpoint中缺少某个域的数据

**解决**: 确保checkpoint完整，或使用正确的checkpoint文件

### 问题3: 加载后性能下降

**原因**: 可能是随机种子、设备或其他环境因素不同

**解决**: 设置相同的随机种子 `--random_seed 42`，使用相同的设备

## 性能提升

使用checkpoint可以显著节省训练时间：

| 场景 | 完整训练时间 | 从KG恢复 | 从KT恢复 |
|------|--------------|----------|----------|
| 4域实验 | ~2小时 | ~30分钟 | ~10分钟 |
| 时间节省 | - | **75%** | **92%** |

## 进阶使用

### 自定义Checkpoint保存逻辑

修改 `main.py` 中的保存条件：

```python
# 只在性能提升时保存
if args.save_checkpoint and hr_10 > best_hr:
    checkpoint_manager.save_kg_checkpoint(...)

# 每N轮保存一次
if args.save_checkpoint and i % 10 == 0:
    checkpoint_manager.save_kg_checkpoint(...)
```

### Checkpoint压缩

在 `checkpoint.py` 中启用压缩：

```python
torch.save(model_states, path, _use_new_zipfile_serialization=True)
```

### 分布式Checkpoint

对于大规模训练，可以分片保存：

```python
# 按域分片保存
for i, server in enumerate(servers):
    shard_path = f"{checkpoint_path}/domain_{i}.pt"
    torch.save(server.state_dict(), shard_path)
```
