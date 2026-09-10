# RealMotion 技术说明文档

## 1. 项目概述

RealMotion 是面向自动驾驶连续场景的多模态运动预测框架，对应论文《Motion Forecasting in Continuous Driving》。项目基于 Argoverse 2 Motion Forecasting 数据集，预测目标车辆未来 6 秒的多条可能轨迹。

传统方法通常把相邻时刻的场景分别推理，RealMotion 则显式利用上一时刻已经计算出的场景特征和预测轨迹，使预测过程具有连续记忆。

项目的核心表达可以概括为：

```text
RealMotion
= 单帧场景编码与多模态预测
+ Scene Context Stream（历史场景特征继承）
+ Agent Trajectory Stream（历史预测轨迹接力）
```

默认输入由三个连续场景帧组成：

```text
t = 3s  →  t = 4s  →  t = 5s
  memory    memory      最终预测
```

每个时刻使用最近 3 秒历史信息，输出未来 6 秒、6 种可能模式的二维轨迹。

---

## 2. 技术目标

项目主要解决以下问题：

1. **单帧预测缺少上下文连续性**：相邻时刻重复编码相似场景，却没有复用历史计算结果。
2. **历史预测未被利用**：上一时刻对未来的判断通常在下一次推理时被完全丢弃。
3. **局部场景随车辆移动而变化**：需要处理不同时间局部坐标系之间的位置、朝向和时间差。
4. **未来存在多种可能性**：模型需要同时输出多条候选轨迹及对应置信度。

RealMotion 通过跨帧场景特征引用、轨迹对齐、交叉注意力和增量修正完成连续预测。

---

## 3. 整体系统结构

```text
Argoverse 2 原始数据
  ├── scenario parquet
  └── static map JSON
          │
          ▼
数据预处理 Av2Extractor
  ├── 110 帧 agent 状态
  ├── 车道中心线及属性
  └── focal agent / 场景元数据
          │
          ▼
保存为 processed/*.pt
          │
          ▼
Av2Dataset 连续场景重组
  ├── 3s 时刻局部场景
  ├── 4s 时刻局部场景
  └── 5s 时刻局部场景
          │
          ▼
RealMotion
  ├── Agent History Encoder
  ├── Lane Encoder
  ├── Scene Context Referencing
  ├── Scene Transformer Encoder
  ├── Multimodal Decoder
  └── Trajectory Relaying
          │
          ▼
输出
  ├── focal agent: 6 × 60 × 2
  ├── mode logits: 6
  └── surrounding agents: (N-1) × 60 × 2
          │
          ├── 训练损失
          ├── 验证指标
          └── AV2 submission.parquet
```

---

## 4. 项目目录与模块职责

```text
RealMotion/
├── train.py
├── eval.py
├── preprocess.py
├── requirements.txt
│
├── conf/
│   ├── config.yaml
│   ├── datamodule/
│   │   ├── av2.yaml
│   │   └── av2_stream.yaml
│   └── model/
│       ├── RealMotion_I.yaml
│       └── RealMotion.yaml
│
└── realmotion/
    ├── datamodules/
    │   ├── av2_data_utils.py
    │   ├── av2_extractor.py
    │   ├── av2_dataset.py
    │   └── av2_datamodule.py
    │
    ├── model/
    │   ├── realmotion.py
    │   ├── pl_modules.py
    │   └── layers/
    │       ├── agent_embedding.py
    │       ├── lane_embedding.py
    │       ├── transformer_blocks.py
    │       ├── multimodal_decoder.py
    │       ├── mtr_decoder.py
    │       └── mln.py
    │
    ├── metrics/
    │   ├── min_ade.py
    │   ├── min_fde.py
    │   ├── mr.py
    │   └── utils.py
    │
    └── utils/
        ├── optim.py
        └── submission_av2.py
```

### 4.1 顶层入口

| 文件 | 职责 |
|---|---|
| `preprocess.py` | 扫描 AV2 原始文件，多进程执行场景预处理 |
| `train.py` | Hydra 配置加载、对象实例化、Lightning 训练与验证 |
| `eval.py` | 加载 checkpoint，执行验证或测试集提交预测 |
| `requirements.txt` | AV2、Hydra、Lightning、Timm、TorchMetrics 等依赖 |

### 4.2 配置层

| 配置 | 作用 |
|---|---|
| `config.yaml` | 全局随机种子、GPU、batch size、epoch、回调、Trainer 参数 |
| `av2.yaml` | 单时刻数据：50 帧历史，`split_points=[50]` |
| `av2_stream.yaml` | 连续数据：30 帧历史，`split_points=[30,40,50]` |
| `RealMotion_I.yaml` | 独立场景基线模型及基础 LightningModule |
| `RealMotion.yaml` | 完整连续模型及 StreamLightningModule |

### 4.3 数据层

| 文件 | 作用 |
|---|---|
| `av2_data_utils.py` | 读取 parquet、静态地图，定义 agent/lane 类型映射 |
| `av2_extractor.py` | 将原始场景转换为统一张量并保存为 `.pt` |
| `av2_dataset.py` | 局部坐标变换、范围过滤、连续帧切分、目标构造和 batch padding |
| `av2_datamodule.py` | 建立训练、验证和测试 DataLoader |

### 4.4 模型层

| 文件 | 作用 |
|---|---|
| `realmotion.py` | 主干模型、两条连续信息流和 memory 更新 |
| `agent_embedding.py` | 基于 Conv1d、Neighborhood Attention 和多尺度融合编码历史轨迹 |
| `lane_embedding.py` | 使用 PointNet 风格的逐点卷积和池化编码车道折线 |
| `transformer_blocks.py` | 当前场景 self-attention、跨时间 cross-attention |
| `mln.py` | 位姿条件化 LayerNorm 和正余弦位置编码 |
| `multimodal_decoder.py` | 生成 6 条未来轨迹及 mode logits |
| `mtr_decoder.py` | 实验性的 Transformer 轨迹解码器，当前主路径未接入 |
| `pl_modules.py` | 损失、训练循环、连续 memory 管理、验证和测试 |

---

## 5. 数据结构与预处理

### 5.1 原始场景张量

一个 AV2 场景被预处理为：

```python
{
    "x_positions":    [N, 110, 2],
    "x_angles":       [N, 110],
    "x_velocity":     [N, 110],
    "x_valid_mask":   [N, 110],
    "x_attr":         [N, 3],
    "lane_positions": [M, 20, 2],
    "lane_attr":      [M, 3],
    "focal_idx":      int,
    "scenario_id":    str,
    "agent_ids":      list
}
```

其中：

- `N` 为 agent 数量；
- `M` 为 lane segment 数量；
- 110 帧由 50 帧历史和 60 帧未来组成；
- 数据采样间隔为 0.1 秒；
- 每条 lane centerline 被插值为 20 个点。

### 5.2 连续场景重组

默认流式数据配置：

```yaml
num_historical_steps: 30
split_points: [30, 40, 50]
```

对应关系：

| 场景帧 | 历史输入 | 预测未来 |
|---|---|---|
| step 30 | 0～3 秒 | 3～9 秒 |
| step 40 | 1～4 秒 | 4～10 秒 |
| step 50 | 2～5 秒 | 5～11 秒 |

每个样本最终返回一个按时间排序的列表，而不是单个字典：

```python
[data_at_3s, data_at_4s, data_at_5s]
```

### 5.3 局部坐标变换

每个场景帧以 focal agent 当前状态为基准：

```text
origin = focal agent 当前坐标
theta  = focal agent 当前朝向
```

agent 和 lane 的全局坐标都会平移并旋转到当前局部坐标系。未来监督轨迹表示为：

```text
target = future_position - current_agent_center
```

局部坐标归一化减少了绝对位置、道路方向和城市坐标系对模型的影响。

### 5.4 数据筛选

数据加载阶段保留：

- 当前时刻状态有效的 agent；
- 距 focal agent 不超过 150 米的 agent 和 lane；
- 距离最近 lane 点小于 5 米的 agent；
- focal agent 无条件保留。

### 5.5 Batch 组织

不同场景中的 agent 和 lane 数量不同，`collate_fn` 使用 `pad_sequence` 补齐，并生成：

```text
x_key_valid_mask    [B, N]
lane_key_valid_mask [B, M]
```

Transformer 使用对应 mask 屏蔽 padding token。

---

## 6. 模型架构

### 6.1 Agent History Encoder

历史轨迹输入通道为：

```text
[Δx, Δy, Δvelocity, valid_mask]
```

输入形状：

```text
[B, N, T, 4]
```

将 batch 和 agent 维合并后，编码流程为：

```text
Conv1d Tokenizer
    ↓
三级 Neighborhood Attention
    ├── 32 维
    ├── 64 维
    └── 128 维
    ↓
多尺度 FPN 式融合
    ↓
取最后一个时间位置
    ↓
actor feature [B, N, 128]
```

Neighborhood Attention 主要捕获局部时间邻域中的运动变化，多尺度融合则结合短期与长期运动特征。

### 6.2 Lane Encoder

每条 lane 的输入为：

```text
[relative_x, relative_y, valid_mask]
```

Lane Encoder 采用 PointNet 风格结构：

```text
逐点 Conv1d
    ↓
第一次 max pooling，提取整条车道的全局特征
    ↓
全局特征与逐点特征拼接
    ↓
第二组 Conv1d
    ↓
第二次 max pooling
    ↓
lane feature [B, M, 128]
```

### 6.3 场景 Token

模型分别为 agent 和 lane 加入类型 embedding，并基于以下四维状态生成位置编码：

```text
[center_x, center_y, cos(angle), sin(angle)]
```

最终得到：

```text
Agent Token = 轨迹特征 + 类型特征 + 位姿特征
Lane Token  = 几何特征 + 类型特征 + 位姿特征
```

二者拼接成统一场景序列：

```text
x_encoder: [B, N + M, 128]
```

### 6.4 Scene Context Stream

Scene Context Stream 将上一时刻场景特征传递给当前时刻。

相邻帧之间首先计算：

```text
memory_pose = [Δtime, Δheading, Δposition_x, Δposition_y]
```

相对位姿经过多频正余弦编码，然后由 MLN 调制当前和历史特征：

```text
MLN(x, pose) = gamma(pose) × LayerNorm(x) + beta(pose)
```

之后执行两种 cross-attention：

```text
当前 agent token → 查询上一帧全部场景 token
当前 lane token  → 查询上一帧 lane token
```

该结构使当前场景能够引用此前已编码的交通参与者、道路结构和交互关系。

### 6.5 当前场景 Transformer

历史上下文增强后的 token 进入四层 Transformer Encoder：

```yaml
embed_dim: 128
encoder_depth: 4
num_heads: 8
mlp_ratio: 4
```

每层采用 Pre-LayerNorm 结构：

```text
LayerNorm → Multi-Head Self-Attention → Residual
LayerNorm → MLP → Residual
```

这里完成当前场景内的 agent-agent、agent-map 和 map-map 交互。

### 6.6 Multimodal Decoder

第 0 个 agent 固定为 focal agent，其场景特征被映射成 6 个 mode feature：

```text
[B, 128] → [B, 6, 128]
```

每个 mode 输出：

```text
轨迹坐标：60 × 2
置信度 logit：1
```

最终输出：

```text
y_hat  [B, 6, 60, 2]
pi     [B, 6]
x_mode [B, 6, 128]
```

周边 agent 使用辅助 dense predictor，每个 agent 输出一条未来轨迹：

```text
y_hat_others [B, N-1, 60, 2]
```

### 6.7 Agent Trajectory Stream

Trajectory Stream 使用上一时刻的预测结果修正当前预测。

处理步骤：

1. 根据时间差定位上一预测中与当前时刻对应的轨迹点；
2. 从上一预测的所有轨迹点中减去该参考点；
3. 将历史轨迹旋转到当前 focal agent 的局部坐标系；
4. 编码当前轨迹和历史轨迹；
5. 当前 6 个 mode 对历史 6 个 mode 执行 cross-attention；
6. 根据增强后的 mode feature 预测轨迹修正量。

最终结果：

```text
final_y_hat = current_y_hat + y_hat_diff
```

这种设计保留了上一时刻对未来趋势的判断，同时允许当前观测对其进行修正。

---

## 7. 时序 Memory 结构

完整 RealMotion 在每次 forward 后返回：

```python
memory_dict = {
    "x_encoder":   current_scene_tokens,
    "x_mode":      current_mode_features,
    "glo_y_hat":   rotated_prediction_displacements,
    "x_mask":      scene_valid_mask,
    "x_type_mask": agent_lane_type_mask,
    "origin":      current_origin,
    "theta":       current_heading,
    "timestamp":   current_timestamp,
}
```

下一帧使用关系：

```text
x_encoder + pose + mask
        └── Scene Context Stream

x_mode + glo_y_hat
        └── Agent Trajectory Stream
```

Memory 只保存上一处理时刻的结果，但上一时刻结果已经吸收更早历史，因此信息可以递归累积。

---

## 8. 训练结构

### 8.1 多模态 Winner-Takes-All

对 6 个预测模式分别计算与 GT 的累计 L2 距离：

```text
best_mode = argmin(sum_t distance(pred_mode, ground_truth))
```

仅最佳模式参与主要轨迹回归：

```text
agent_reg_loss = SmoothL1(best_trajectory, ground_truth)
```

同时使用交叉熵训练模式概率：

```text
agent_cls_loss = CrossEntropy(pi, best_mode)
```

该策略避免所有模式收敛到相似的平均轨迹。

### 8.2 周边 Agent 辅助监督

对有效的周边 agent 未来位置计算：

```text
others_reg_loss = SmoothL1(predicted_others, target_others)
```

总损失为：

```text
loss = agent_reg_loss
     + agent_cls_loss
     + others_reg_loss
```

### 8.3 连续帧训练

`StreamLightningModule` 按时间依次处理场景：

```text
memory = None
for frame in sequence:
    frame.memory = memory
    output = model(frame)
    loss += calculate_loss(output)
    memory = output.memory
```

`num_grad_frame` 控制保留梯度的最后若干帧。更早的帧可以在 `no_grad` 模式下仅更新 memory，从而控制跨时间计算图大小。

默认三个输入帧且 `num_grad_frame=3`，所以三帧均参与反向传播。

---

## 9. 验证指标

| 指标 | 含义 |
|---|---|
| `minADE1` | 概率最高轨迹的平均位移误差 |
| `minADE6` | 6 条候选中最优轨迹的平均位移误差 |
| `minFDE1` | 概率最高轨迹的最终点误差 |
| `minFDE6` | 6 条候选中最优轨迹的最终点误差 |
| `MR` | 6 条候选终点是否全部超过 2 米误差 |
| `b-minFDE6` | minFDE 加模式概率校准惩罚 |

项目默认根据 `minADE6` 保存表现最好的 checkpoint。

---

## 10. 优化与运行配置

默认训练设置：

```yaml
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-2
min_learning_rate: 1e-5
epochs: 80
warmup_ratio: 0.167
batch_size: 32
gpus: 4
gradient_clip: 5
```

学习率策略：

```text
线性 Warmup → Cosine Decay → min_lr
```

Linear、Conv 和 Attention 权重使用 weight decay；bias、LayerNorm 和 Embedding 参数不使用 weight decay。

---

## 11. RealMotion 与 RealMotion-I

| 对比项 | RealMotion-I | RealMotion |
|---|---|---|
| 输入 | 单个场景帧 | 多个连续场景帧 |
| Scene Context Stream | 无 | 有 |
| Trajectory Stream | 无 | 有 |
| Memory | 无 | 有 |
| LightningModule | `BaseLightningModule` | `StreamLightningModule` |
| 默认数据配置 | `av2.yaml` | `av2_stream.yaml` |

单帧基线运行方式：

```bash
python train.py model=RealMotion_I datamodule=av2
```

完整连续模型：

```bash
python train.py
```

---

## 12. 训练、验证与提交

### 数据预处理

```bash
python preprocess.py -d /path/to/data -p
```

### 训练

```bash
python train.py
```

### 恢复训练

```bash
python train.py checkpoint=/path/to/checkpoint.ckpt
```

### 验证

```bash
python eval.py checkpoint=/path/to/checkpoint.ckpt
```

### 生成 AV2 测试提交

```bash
python eval.py checkpoint=/path/to/checkpoint.ckpt submit=true
```

测试输出会从 focal agent 局部坐标转换回全局坐标，并保存为 AV2 ChallengeSubmission 所需的 parquet 格式。

---

## 13. 核心技术点总结

### 数据层

1. 将 AV2 独立 benchmark 场景重组为连续时间序列。
2. 使用滑动历史窗口构造 3s、4s、5s 三个连续观测帧。
3. 所有场景转换到 focal agent 局部坐标系。
4. 使用位置差、速度差和有效 mask 表达 agent 历史运动。
5. 使用 padding mask 支持不同数量的 agent 和 lane。

### 表征层

1. Neighborhood Attention 处理 agent 时间序列。
2. 多尺度结构融合不同时间感受野。
3. PointNet 风格网络编码 lane polyline。
4. 类型 embedding 区分 agent 与地图元素。
5. 位置、朝向共同构成场景 token 的空间编码。

### 交互层

1. 当前场景 Transformer 建模 agent-map 空间交互。
2. Scene Context Stream 对历史场景 token 执行 cross-attention。
3. 使用相对时间、位置和朝向对跨帧特征进行对齐条件化。
4. MLN 根据相对位姿动态调整特征分布。

### 预测层

1. 固定输出 6 条多模态候选轨迹。
2. 使用 mode classification 表达轨迹概率。
3. Trajectory Stream 对齐并继承上一时刻预测。
4. 通过 residual refinement 修正当前基础轨迹。
5. 周边 agent 预测作为辅助监督改善场景理解。

### 训练层

1. Winner-Takes-All 选择最接近 GT 的预测模式。
2. Smooth L1 负责轨迹回归，交叉熵负责模式分类。
3. 支持 memory 跨帧梯度传播。
4. 支持仅对最后若干连续帧保留梯度，控制显存。
5. AdamW 配合 warmup 和 cosine learning-rate schedule。

---

## 14. 工程实现注意事项

当前公开代码包含一些未完全接通或需要增强的部分：

1. `use_stream_encoder` 与 `use_stream_decoder` 被保存但没有实际控制 forward 分支，无法直接用于单流消融。
2. `mtr_decoder.py` 中的 TransformerDecoder 已定义，但主模型 forward 没有调用该分支。
3. 损失函数保留 `new_y_hat` 接口，但当前模型不会输出该字段。
4. `preprocess.py` 的 `--batch` 参数被读取但未实际使用。
5. `Av2Extractor.save()` 在提取失败后仍可能尝试保存未初始化的 `data`。
6. focal agent 的未来损失和指标默认未来 60 帧全部有效，没有显式应用 future mask。
7. Trajectory Stream 修正轨迹后，没有重新计算对应的 mode logits。
8. `glo_y_hat` 实际更接近旋转到全局方向的相对位移，而不是已经加回原点的完整全局坐标。

这些问题不改变核心方法，但在复现、消融实验或迁移到其他数据集时需要特别检查。

---

## 15. 总体评价

RealMotion 的主要贡献并不是重新设计一个复杂的单帧轨迹预测骨干，而是把轨迹预测从“独立场景推理”改造成“连续状态更新”：

```text
当前观测
  + 历史场景语义
  + 历史未来判断
  = 当前连续轨迹预测
```

从工程视角看，整个项目分为五层：

```text
数据预处理层
    ↓
连续场景组织层
    ↓
Agent / Lane 表征层
    ↓
场景与轨迹双流交互层
    ↓
多模态预测、训练和评估层
```

其中最关键的代码阅读顺序是：

1. `av2_dataset.py`：理解连续场景如何产生；
2. `realmotion.py`：理解模型总数据流和 memory；
3. `transformer_blocks.py`：理解跨帧交互及位姿对齐；
4. `agent_embedding.py` 与 `lane_embedding.py`：理解输入表征；
5. `pl_modules.py`：理解损失、跨帧训练和评估。
