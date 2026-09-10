# QCNet 技术说明文档与核心技术点总结

分析日期：2026-09-10  
源码仓库：[ZikangZhou/QCNet](https://github.com/ZikangZhou/QCNet)  
分析版本：`55cacb418cbbce3753119c1f157360e66993d0d0`（本次下载的 main 快照）  
分析方式：静态阅读主要源码与调用关系，未执行数据集训练或精度复现。

## 1. 项目定位

QCNet 是 CVPR 2023 论文《Query-Centric Trajectory Prediction》的官方实现。模型使用交通参与者的历史运动状态和矢量地图，为目标交通参与者预测多种可能的未来轨迹及其概率。

核心技术由三部分组成：

1. **Query-centric 几何表达**：以接收信息的节点为参考，使用相对距离、相对方位、朝向差和时间差描述关系。
2. **场景图编码**：通过地图点到道路片段、道路片段之间、历史时间之间、地图到 agent、agent 之间的注意力聚合信息。
3. **两阶段轨迹解码**：先循环生成候选轨迹，再将整条候选轨迹编码成 query，预测修正量与模式概率。

标准 AV2 设置使用 50 帧历史预测 60 帧未来，采样间隔为 0.1 秒，对应 5 秒历史和 6 秒未来。默认每个 agent 输出 6 个模式。

### 1.1 公开实现的范围

当前公开主路径是 Argoverse 2 的边缘轨迹预测。模型编码多个 agent，并为 agent 产生各自的候选轨迹；验证指标和测试提交选择 `category == 3` 的 focal agent。

不同 agent 的同一模式序号不应直接解释为一个联合场景假设。README 提到的联合预测扩展 QCNeXt，也不应与本仓库已经接通的 QCNet 训练和提交路径混为一谈。

## 2. 总体结构

```text
AV2 parquet + 地图 JSON
           │
           ▼
ArgoverseV2Dataset
  ├── agent 历史/未来状态与 mask
  ├── map_point 细粒度几何
  ├── map_polygon 道路片段
  └── 静态地图拓扑边
           │
           ▼
HeteroData + TargetBuilder + PyG Batch
           │
           ▼
QCNetEncoder
  ├── QCNetMapEncoder
  │     └── point → polygon → polygon
  └── QCNetAgentEncoder
        └── temporal → polygon-to-agent → agent-to-agent
           │
           ▼
QCNetDecoder
  ├── 可学习 mode queries
  ├── 循环生成轨迹 Proposal
  ├── Fourier + GRU 编码完整 Proposal
  └── Refinement + 分布尺度 + 模式 logits
           │
           ├── 训练：两阶段 NLL + 混合分布 NLL
           ├── 验证：位移、朝向、概率和漏预测指标
           └── 测试：转换全局坐标，写入 AV2 parquet
```

### 2.1 软件分层

| 层级 | 代码位置 | 职责 |
|---|---|---|
| 运行入口 | `train_qcnet.py`、`val.py`、`test.py` | 参数解析、模型加载、Trainer 调用 |
| 数据层 | `datasets/`、`datamodules/`、`transforms/` | 原始数据处理、图构建、GT 坐标变换和批处理 |
| 模型组织层 | `predictors/qcnet.py` | 组合编码器和解码器，定义损失、评估与优化器 |
| 网络模块层 | `modules/` | 地图编码、agent 编码和两阶段解码 |
| 通用算子层 | `layers/` | 图注意力、Fourier 编码和 MLP |
| 训练辅助层 | `losses/`、`metrics/`、`utils/` | 分布损失、指标、几何和图工具 |

## 3. 目录与文件职责

```text
QCNet/
├── train_qcnet.py
├── val.py
├── test.py
├── environment.yml
├── datasets/
│   └── argoverse_v2_dataset.py
├── datamodules/
│   └── argoverse_v2_datamodule.py
├── transforms/
│   └── target_builder.py
├── predictors/
│   └── qcnet.py
├── modules/
│   ├── qcnet_encoder.py
│   ├── qcnet_map_encoder.py
│   ├── qcnet_agent_encoder.py
│   └── qcnet_decoder.py
├── layers/
│   ├── attention_layer.py
│   ├── fourier_embedding.py
│   └── mlp_layer.py
├── losses/
├── metrics/
└── utils/
```

| 文件 | 核心类/功能 | 阅读重点 |
|---|---|---|
| `datasets/argoverse_v2_dataset.py` | `ArgoverseV2Dataset` | `get_agent_features()`、`get_map_features()`、缓存与完整性检查 |
| `datamodules/argoverse_v2_datamodule.py` | `ArgoverseV2DataModule` | `prepare_data()`、`setup()`、PyG DataLoader |
| `transforms/target_builder.py` | `TargetBuilder` | 每个 agent 的局部未来位置和相对朝向 |
| `predictors/qcnet.py` | `QCNet` | `forward()`、`training_step()`、`validation_step()`、`test_step()` |
| `modules/qcnet_encoder.py` | `QCNetEncoder` | 先编码地图，再编码 agent |
| `modules/qcnet_map_encoder.py` | `QCNetMapEncoder` | pt2pl 与 pl2pl 注意力 |
| `modules/qcnet_agent_encoder.py` | `QCNetAgentEncoder` | 时间、地图、交互图边及关系编码 |
| `modules/qcnet_decoder.py` | `QCNetDecoder` | mode query、分段 proposal、GRU、残差 refinement |
| `layers/attention_layer.py` | `AttentionLayer` | 关系感知 Key/Value、邻居 softmax 和门控 |
| `layers/fourier_embedding.py` | `FourierEmbedding` | 可学习频率、逐维 MLP、类别特征融合 |
| `losses/nll_loss.py` | `NLLLoss` | 不同输出维度对应的概率分布 |
| `losses/mixture_nll_loss.py` | `MixtureNLLLoss` | 模式概率的混合分布似然 |
| `metrics/` | 位移、朝向和概率指标 | 候选选择标准与有效步过滤 |
| `utils/` | 几何、图边和初始化工具 | 角度归一化、边合并、稀疏图转换 |

## 4. 数据流与张量约定

本文采用：`A` 表示一个 batch 内的 agent 总数，`M` 表示 polygon 总数，`P` 表示 map point 总数，`H=50`、`F=60`、`K=6`、`D=128`。

### 4.1 数据处理生命周期

数据集读取场景 parquet 与地图 JSON，提取 agent 和地图信息，保存到每个场景对应的 `.pkl` 文件。加载时用 `HeteroData` 包装字典。默认目录为：

```text
dataset_root/
├── train/
│   ├── raw/<scenario_id>/
│   └── processed/*.pkl
├── val/
│   ├── raw/<scenario_id>/
│   └── processed/*.pkl
└── test/
    ├── raw/<scenario_id>/
    └── processed/*.pkl
```

数据下载、处理和加载由 Dataset/DataModule 管理，没有单独的顶层 preprocess 脚本。

### 4.2 异构图节点

```python
data['agent']
data['map_point']
data['map_polygon']
data['map_point', 'to', 'map_polygon']
data['map_polygon', 'to', 'map_polygon']
```

| Agent 字段 | 形状 | 说明 |
|---|---|---|
| `position` | `[A, H+F, dim]` | 位置；AV2 agent 实际填充 XY |
| `heading` | `[A, H+F]` | 朝向 |
| `velocity` | `[A, H+F, dim]` | 速度向量 |
| `valid_mask` | `[A, H+F]` | 状态/历史向量是否有效 |
| `predict_mask` | `[A, H+F]` | 哪些未来步参与预测监督 |
| `type` | `[A]` | 10 类对象类型 |
| `category` | `[A]` | fragment、unscored、scored、focal |
| `target` | `[A, F, 4]` | Transform 构造的局部 XYZ 与相对 heading |

默认过滤历史窗口从未出现的 agent。启用默认 `vector_repr=True` 时，历史状态 t 需要 t 和 t-1 均有效，才能形成有效运动向量。`predict_mask` 与 `valid_mask` 分工不同，前者用于监督，后者主要控制历史编码。

PyG 把不同场景节点拼接，通过 `batch` 与 `ptr` 标记所属场景。空间构图时利用场景和时间编号隔离图，避免跨场景或跨时间错误连边。时间轴仍然固定长度，并使用 mask 处理缺失状态。

### 4.3 地图的两级表示

`map_point` 表示车道左边界、右边界和中心线上小线段的起点，并携带方向、长度、线型和侧别。因此它包含有向局部几何，而不是只有坐标的孤立点。

`map_polygon` 表示 lane segment 或有向人行横道。车道参考位置取中心线起点，方向取第一段方向。每个人行横道构造成两个相反方向的 polygon。

地图拓扑包括前驱 PRED、后继 SUCC、左邻 LEFT 和右邻 RIGHT；编码阶段还加入半径内的 polygon 空间邻接边。

### 4.4 监督目标坐标系

`TargetBuilder` 为每个 agent 使用最后历史时刻的位置和朝向建立局部坐标系：

```text
local_future_xy = (global_future_xy - current_xy) × rotation
relative_heading = wrap(future_heading - current_heading)
```

默认训练只使用二维位置；开启 `output_head` 后还监督朝向。场景输入不需要为每个预测对象重复生成一整份旋转后的场景，编码器通过 query-relative 关系表达几何。

## 5. Query-centric 技术原理

对于接收消息的节点 i，邻居 j 的关系采用：

```text
r(j→i) = [距离，邻居方位相对 i 朝向的夹角，邻居与 i 的朝向差]
```

时间交互追加相对时间步。Query 可以是 polygon、某历史时刻的 agent 状态或解码 mode。

这使编码依赖相对空间关系而非世界坐标原点；整体场景平移和旋转不改变这些关系。局部预测转换回世界坐标后随场景变换，表现为世界坐标输出的相应等变性。

时间使用相对时间差，避免把绝对时间编号作为语义。该设计有利于潜在的历史特征复用，但当前公开 forward 没有实现跨调用缓存管理。

### 5.1 FourierEmbedding

连续输入 x 经可学习频率 w 编码为：

```text
[cos(2πwx), sin(2πwx), x]
```

各连续维度分别经过 MLP 后求和，再加入类别 embedding，输出统一 D 维特征。默认每个输入维度使用 64 个频率。

### 5.2 AttentionLayer

注意力只在 `edge_index` 中定义的边上计算。相对几何 r 同时加入 Key 和 Value：

```text
attention(i,j) ∝ exp(q_i · (k_j + W_k r_ij) / sqrt(head_dim))
message_i = Σ_j attention(i,j) × (v_j + W_v r_ij)
```

随后通过 sigmoid 门控融合邻居消息和自身投影，并经过残差、归一化与前馈网络。`bipartite=True` 支持不同源/目标节点类型，例如 polygon-to-agent。

## 6. 编码器结构

### 6.1 QCNetMapEncoder

每层依次执行：

```text
map_point → polygon 注意力：聚合边界和中心线几何
polygon → polygon 注意力：聚合道路拓扑与空间邻居
```

默认一层地图编码。输出 `x_pt[P,D]` 和 `x_pl[M,H,D]`。静态地图特征计算后复制到时间维，供各历史时刻 agent 查询，并非每个时间步重新编码地图。

### 6.2 QCNetAgentEncoder

每个历史状态的初始连续特征为：

1. 相邻位置差的长度；
2. 位移方向相对自身朝向的夹角；
3. 速度大小；
4. 速度方向相对自身朝向的夹角。

经 FourierEmbedding 并加入对象类型后，形成 `x_a[A,H,D]`。

每层依次执行：

| 关系 | 接收者 | 来源 | 目的 |
|---|---|---|---|
| Temporal | 某 agent 的历史时刻状态 | 同一 agent 更早状态 | 建模运动趋势 |
| pl2a | Agent 状态 | 同时刻附近地图 | 建模道路约束 |
| a2a | Agent 状态 | 同时刻附近 agent | 建模交通交互 |

时间边只从过去指向后续时刻，单层跨度由 `time_span` 限制。默认 agent 编码层数为 2，多层可以扩大有效感受野。空间关系通过半径构图，代码设置 `max_num_neighbors=300`。

## 7. 两阶段解码器

### 7.1 Query 与关系命名

每个 agent 使用 K 个可学习 mode embedding 初始化 query。embedding 在 agent 间共享，通过读取各自上下文形成不同预测。

| 名称 | 含义 |
|---|---|
| t2m | Agent 历史时间状态 → mode |
| pl2m | 邻近地图 polygon → mode |
| a2m | 邻近 agent 当前状态 → mode |
| m2m | 同一 agent 的 mode 之间交互 |

`m2m` 不表示所有 agent 候选轨迹之间的联合场景建模。

### 7.2 Proposal：循环生成未来片段

每轮更新 mode query，依次执行 t2m、pl2m、a2m 和 m2m，再预测一段未来位移增量及尺度。标准设置 `F=60`、循环次数为 3，每轮输出 20 步。query 特征在循环间持续更新。

拼接各段后沿时间累加位移增量，得到完整候选轨迹。尺度通过正值变换和累加构造。

此循环发生在一次预测内部，输入场景未随循环加入新观测；场景特征、关系和邻接边在循环前构建。它与 RealMotion 跨观测时刻传递 memory 的含义不同。

### 7.3 Refinement：编码完整轨迹并精修

```text
Proposal 坐标.detach()
        ↓
FourierEmbedding
        ↓
GRU 沿未来时间轴编码
        ↓
每条候选对应一个 trajectory query
        ↓
t2m → pl2m → a2m → m2m
        ↓
完整未来轨迹修正量 + 尺度 + pi
```

精修结果为：

```text
refined_trajectory = proposal.detach() + residual
```

GRU 编码的是生成的未来候选轨迹。Detach 阻止精修损失通过 proposal 坐标直接反传，但两阶段仍可通过各自路径训练共享场景编码器。

### 7.4 输出接口

| 字段 | 形状 | 含义 |
|---|---|---|
| `loc_propose_pos` | `[A,K,F,2]` | 初始位置轨迹 |
| `scale_propose_pos` | `[A,K,F,2]` | 初始位置分布尺度 |
| `loc_refine_pos` | `[A,K,F,2]` | 精修位置轨迹 |
| `scale_refine_pos` | `[A,K,F,2]` | 精修位置分布尺度 |
| `pi` | `[A,K]` | 模式 logits，softmax 后为概率 |
| `loc_*_head`、`conc_*_head` | `[A,K,F,1]` | 可选朝向与集中度参数；未启用时为零占位 |

## 8. 训练损失与优化

### 8.1 最佳模式分配

根据 proposal 与 GT 在有效未来时间步上的累计 L2 距离选择 best mode。Proposal 和 refinement 使用同一个模式索引进行回归监督。

### 8.2 两阶段回归 NLL

位置采用 Laplace 分布，单坐标负对数似然为：

```text
NLL(y; μ,b) = log(2b) + |y-μ|/b
```

模型同时学习位置 μ 和尺度 b。可选朝向使用周期分布 Von Mises。回归损失应用有效未来 mask，并按每个时间步的有效 agent 数量归一化后做时间平均。

### 8.3 模式概率损失

概率分支使用精修终点的混合分布 NLL：

```text
L_cls = -log Σ_k softmax(pi)_k × p(GT_final | mode k)
```

这里使用所有候选的终点分布，分布参数 detach，logits 保留梯度。它不是对 best_mode 的简单硬标签交叉熵。

总损失：

```text
L = L_proposal + L_refine + L_cls
```

### 8.4 优化策略

AdamW 默认学习率 `5e-4`、weight decay `1e-4`；bias、归一化和 embedding 等参数不做权重衰减。学习率使用 CosineAnnealingLR，默认 `T_max=64`，没有在该路径中实现 warmup。Trainer 默认训练 64 个 epoch，根据 `val_minFDE` 保存前 5 个 checkpoint。

## 9. 验证与测试

验证和提交选择 `category == 3` 的 focal agent。训练回归则按 `predict_mask` 选择有效 agent/未来步，不限于 focal agent。

| 指标 | 当前实现含义 |
|---|---|
| minFDE | 候选中最小最终有效位置误差 |
| minADE | 默认先按 FDE 选择候选，再算该候选的有效步 ADE |
| minAHE / minFHE | 平均/最终朝向误差 |
| MR | 候选是否未能命中阈值范围 |
| Brier | 所选最佳候选的概率惩罚 `(1-p)^2` |

`val_Brier` 本身不包含 FDE，不能直接当作 brier-minFDE。未启用朝向输出时，朝向指标使用相邻预测位置差推导方向。

测试取 `loc_refine_pos`，旋转并平移回各 focal agent 对应的世界坐标，再将 `pi` softmax，生成 ChallengeSubmission parquet。

## 10. 关键配置与运行入口

| 参数 | 默认值/示例值 | 作用 |
|---|---|---|
| hidden_dim | 128 | 特征维度 |
| num_modes | 6 | 轨迹模式数 |
| num_map_layers | 1 | 地图层数 |
| num_agent_layers | 2 | Agent 时空层数 |
| num_dec_layers | 2 | 解码交互层数 |
| num_heads / head_dim | 8 / 16 | 多头注意力结构 |
| num_freq_bands | 64 | 每维可学习频率数 |
| dropout | 0.1 | Dropout |
| num_recurrent_steps | 示例 3 | Proposal 循环次数 |
| time_span | 示例 10 | 单层时间注意力跨度 |
| num_t2m_steps | 示例 30 | Mode 读取的最近历史步数 |
| pl2pl_radius | 示例 150m | 地图空间邻接半径 |
| pl2a_radius / a2a_radius | 示例 50m / 50m | 编码器交互半径 |
| pl2m_radius / a2m_radius | 示例 150m / 150m | 解码器上下文半径 |

训练参数由 argparse 管理。以下为 README 训练参数组合，未在本次分析中执行：

```bash
python train_qcnet.py --root /path/to/dataset_root --train_batch_size 4 --val_batch_size 4 --test_batch_size 4 --devices 8 --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150
```

```bash
python val.py --model QCNet --root /path/to/dataset_root --ckpt_path /path/to/model.ckpt
python test.py --model QCNet --root /path/to/dataset_root --ckpt_path /path/to/model.ckpt
```

项目提供的 `environment.yml` 是包含 Linux 构建标记的原始环境快照，主要版本包括 Python 3.8、PyTorch 2.0.1、PyG 2.3.0、Lightning 2.0.4、CUDA 11.8。它不是跨操作系统的最小依赖清单。

## 11. 核心技术点速查

| 技术点 | 代码落点 | 解决的问题 |
|---|---|---|
| Query-centric 关系表达 | 各 encoder/decoder 中的 `r_*` | 消除对绝对坐标系的依赖 |
| 可学习 Fourier 编码 | `fourier_embedding.py` | 表达距离、角度、速度、时间的非线性影响 |
| 地图分层图表示 | Dataset + MapEncoder | 同时表达细粒度几何与道路拓扑 |
| 稀疏邻域注意力 | `radius`、`radius_graph`、AttentionLayer | 控制交互范围并避免全局全连接 |
| 分解式时空交互 | AgentEncoder | 依次融合时间运动、道路约束、邻车交互 |
| 相对关系注入 Key/Value | AttentionLayer | 让几何影响关注权重与消息内容 |
| 多模态 query | Decoder `mode_emb` | 为多个未来假设提供独立表示 |
| 循环分段 Proposal | Decoder recurrent loop | 逐段形成长时域候选轨迹 |
| 轨迹作为 query | Fourier + GRU | 让 refinement 感知完整候选形状 |
| 残差精修 | `loc_refine_pos` | 在初始候选上学习全轨迹修正 |
| 概率分布回归 | Laplace / Von Mises NLL | 同时建模预测值与不确定性参数 |
| 混合终点似然 | MixtureNLLLoss | 学习模式概率 |

## 12. 工程约束与扩展注意事项

以下结论来自静态源码，不代表已经运行触发故障。

1. **修改时间长度要同步数据层。** DataModule 默认 `TargetBuilder(50,60)`，Dataset 也使用默认时间长度；模型的额外参数没有自动透传到这些构造过程。只改 CLI 的历史/未来长度会产生不一致。
2. **子集实验需要修改完整性检查。** Dataset 使用固定官方样本数；文件数不足可能触发下载。下载代码包含删除并重建 split/raw 目录的逻辑，不能把现有小样本目录直接当成已支持的子集模式。
3. **训练会准备三个 split。** `prepare_data()`、`setup()` 都创建 train、val、test。仅训练时也可能触发测试数据准备。
4. **未来长度应能被循环次数整除。** 分段输出大小使用整除并按完整轨迹 reshape，标准组合是 60/3。
5. **多设备测试需要结果汇总。** 当前各进程维护自身 `test_predictions` 并写文件，未见显式跨进程汇总和唯一写出；默认单设备更符合现有提交流程。
6. **Streaming 是架构潜力，不是已完成的在线接口。** 当前 forward 会重新编码窗口，未提供跨调用 memory/cache 管理。
7. **指标定义要核对。** minADE 默认按 FDE 选轨迹；Brier 日志只包含概率项。跨仓库对比需对齐定义。
8. **模式交互不等于联合预测。** 当前 m2m 在每个 agent 内连接候选，不能据此宣称已学习场景级联合模式。

## 13. 与 RealMotion 的结构对照

| 维度 | QCNet | RealMotion |
|---|---|---|
| 数据/交互组织 | PyG 异构图与邻域边 | Agent/lane token 与 padding |
| 坐标表达 | 各 query 的相对几何关系 | Focal 局部坐标与跨帧位姿条件化 |
| 历史编码 | 时间、地图、agent 图注意力 | Neighborhood Attention 与场景 Transformer |
| 解码 | 循环 proposal + GRU + refinement | 基础多模态 MLP + 历史轨迹修正 |
| 循环含义 | 单次预测内生成未来片段 | 连续观测帧之间传递 memory |
| 位置监督 | Laplace NLL | Smooth L1 |
| 概率监督 | 精修终点混合 NLL | 最佳模式交叉熵 |

## 14. 推荐阅读与修改路径

按以下顺序阅读可减少在张量 reshape 和边索引之间反复跳转：

1. `predictors/qcnet.py`：先建立 forward 和损失的整体认识。
2. `datasets/argoverse_v2_dataset.py`、`target_builder.py`：明确图节点、mask 和坐标系。
3. `qcnet_map_encoder.py`：理解点到道路片段的聚合。
4. `qcnet_agent_encoder.py`：追踪三类交互与时空布局变换。
5. `qcnet_decoder.py`：追踪 mode feature 在 proposal 和 refinement 中的变化。
6. `attention_layer.py`、`fourier_embedding.py`：理解关系如何参与消息传递。
7. `losses/`、`metrics/`：核对分布、mask、归一化和候选选择规则。

研究改动应与模块对应：改地图表示从 Dataset/MapEncoder 入手，改交互范围从构图参数入手，改多模态生成从 Decoder 入手，改不确定性建模需同时调整输出头和 NLL，改为在线推理则需要设计缓存有效性与增量图更新。

## 15. 源码参考

- [仓库与 README](https://github.com/ZikangZhou/QCNet)
- [数据集实现](https://github.com/ZikangZhou/QCNet/blob/55cacb418cbbce3753119c1f157360e66993d0d0/datasets/argoverse_v2_dataset.py)
- [主模型与训练逻辑](https://github.com/ZikangZhou/QCNet/blob/55cacb418cbbce3753119c1f157360e66993d0d0/predictors/qcnet.py)
- [地图编码器](https://github.com/ZikangZhou/QCNet/blob/55cacb418cbbce3753119c1f157360e66993d0d0/modules/qcnet_map_encoder.py)
- [Agent 编码器](https://github.com/ZikangZhou/QCNet/blob/55cacb418cbbce3753119c1f157360e66993d0d0/modules/qcnet_agent_encoder.py)
- [两阶段解码器](https://github.com/ZikangZhou/QCNet/blob/55cacb418cbbce3753119c1f157360e66993d0d0/modules/qcnet_decoder.py)
- [图注意力](https://github.com/ZikangZhou/QCNet/blob/55cacb418cbbce3753119c1f157360e66993d0d0/layers/attention_layer.py)
- [Fourier 编码](https://github.com/ZikangZhou/QCNet/blob/55cacb418cbbce3753119c1f157360e66993d0d0/layers/fourier_embedding.py)
