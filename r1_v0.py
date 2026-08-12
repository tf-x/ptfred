import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.mlp import Mlp


class AgentPredictor(nn.Module):
    def __init__(self, hidden_dim, future_len, num_behaviors=4):
        super().__init__()

        self.behavior_estimator = BehaviorEstimation(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            num_behaviors=num_behaviors,
        )
        self.motion_estimator = MotionEstimation(
            in_channels=hidden_dim,
            horizon=future_len,
            hidden_dim=hidden_dim,
        )

    def forward(self, agent_feats, agent_ctrs):
        behavior_logits, behavior_emb = self.behavior_estimator(agent_feats)
        agent_trajs, agent_vel = self.motion_estimator(agent_feats, agent_ctrs, behavior_emb)
        return {
            'agent_trajs': agent_trajs,          # [bs, A, horizon, 2]  绝对坐标
            'agent_vel': agent_vel,              # [bs, A, horizon]     逐帧速度(由delta范数得到)
            'behavior_logits': behavior_logits,  # [bs, A, num_behaviors]
        }


class BehaviorEstimation(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_behaviors=4):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_behaviors),
        )
        self.behavior_embed = nn.Embedding(num_behaviors, hidden_dim)

    def forward(self, agent_feats):
        behavior_logits = self.cls_head(agent_feats)
        behavior_probs = torch.softmax(behavior_logits, dim=-1)
        behavior_emb = behavior_probs @ self.behavior_embed.weight
        return behavior_logits, behavior_emb


class MotionEstimation(nn.Module):
    """
    输出逐帧位移 delta_t = (dx_t, dy_t)，再 cumsum 得到绝对轨迹。
    delta 是原始 xy 增量，不做任何"车头朝向为正方向"的假设，
    因此对向行驶（dx<0 或朝向与自车相反）、侧向/横穿轨迹都能正常表达——
    速度用 delta 的 L2 范数计算，是旋转不变量，不依赖行驶方向。
    """
    def __init__(self,
                 in_channels,
                 horizon=30,
                 hidden_dim=64,
                 use_conv=False):
        super().__init__()
        self.in_channels = in_channels
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        self.feat_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_dim * 2,
            out_features=hidden_dim,
            drop=0.0,
            use_conv=use_conv,
            act_layer=nn.ReLU,
        )

        # FiLM: behavior_emb -> (gamma, beta)
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)

        # 逐帧 delta 解码：用 GRUCell 显式建模时间依赖，
        # 使"某一帧减速"能立刻反映在该帧输出的 delta 上，
        # 而不是像一次性 MLP 那样把整条轨迹一起回归、局部突变被平均掉
        self.step_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),   # -> (dx_t, dy_t)
        )
        self.step_query = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, agent_feats, agent_ctrs, behavior_emb):
        bs, A = agent_feats.shape[:2]

        feat = self.feat_proj(agent_feats)                     # [bs, A, hidden]
        gamma, beta = self.film(behavior_emb).chunk(2, dim=-1)
        cond_feat = feat * (1.0 + gamma) + beta                 # 行为条件注入

        hidden = cond_feat.reshape(bs * A, -1)
        step_in = self.step_query.unsqueeze(0).expand(bs * A, -1)

        deltas = []
        for _ in range(self.horizon):
            hidden = self.step_cell(step_in, hidden)
            delta_t = self.delta_head(hidden)                   # [bs*A, 2]
            deltas.append(delta_t)
            step_in = hidden                                     # 用当前 hidden 驱动下一步，形成时间依赖

        deltas = torch.stack(deltas, dim=1).reshape(bs, A, self.horizon, 2)

        agent_trajs = agent_ctrs + torch.cumsum(deltas, dim=2)   # 逐帧累积得到绝对坐标

        # 速度 = delta 范数 / dt，方向无关，天然支持任意行驶方向（含对向来车）
        dt = 0.08
        agent_vel = torch.norm(deltas, dim=-1) / dt              # [bs, A, horizon]

        return agent_trajs, agent_vel


import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentLoss(nn.Module):
    """
    配合 delta+cumsum 版 MotionEstimation 的损失：
    - gt_behavior 由 GT 未来速度变化自动生成，无需人工标注
    - delta 回归损失：直接监督逐帧位移向量，捕捉突然减速/转向，且不受 cumsum 累积误差污染
    - position 回归损失：监督最终绝对轨迹精度（会受累积误差影响，但反映实际部署时的真实误差）
    - 速度损失：用 delta 范数计算，方向无关，天然支持对向/任意方向行驶
    - 减速片段全部按帧加权（delta/position/velocity 共用同一 weight，保持一致）
    """
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.dt = config.get('dt', 0.1)

        self.delta_weight = config.get('delta_weight', 1.0)
        self.reg_weight = config.get('reg_weight', 1.0)
        self.vel_weight = config.get('vel_weight', 1.0)
        self.behavior_weight = config.get('behavior_weight', 1.0)

        self.decel_weight = config.get('decel_weight', 2.0)          # 减速帧加权系数
        self.decel_accel_thresh = config.get('decel_accel_thresh', -1.0)

        # gt_behavior 判定阈值
        self.brake_accel_thresh = config.get('brake_accel_thresh', -3.0)
        self.decel_label_thresh = config.get('decel_label_thresh', -0.5)
        self.accel_label_thresh = config.get('accel_label_thresh', 0.5)
        self.stop_vel_thresh = config.get('stop_vel_thresh', 1.0)

    def _gt_kinematics(self, gt_traj, gt_mask):
        """
        gt_delta: 逐帧位移向量 (dx,dy)，与预测 deltas 直接对应
        gt_vel/accel: 由 delta 范数算出的标量速度/加速度，方向无关
        """
        gt_delta = torch.zeros_like(gt_traj)
        gt_delta[:, :, 1:] = gt_traj[:, :, 1:] - gt_traj[:, :, :-1]
        gt_delta[:, :, 0] = gt_delta[:, :, 1]          # 首帧用第二帧位移填充，避免为0

        gt_vel = torch.norm(gt_delta, dim=-1) / self.dt
        gt_accel = torch.zeros_like(gt_vel)
        gt_accel[:, :, 1:] = (gt_vel[:, :, 1:] - gt_vel[:, :, :-1]) / self.dt
        return gt_delta, gt_vel, gt_accel

    def _generate_gt_behavior(self, gt_vel, gt_accel, gt_mask):
        valid_len = gt_mask.sum(dim=-1).clamp(min=1.0)
        avg_accel = (gt_accel * gt_mask).sum(dim=-1) / valid_len

        last_idx = (valid_len - 1).clamp(min=0).long()
        end_vel = torch.gather(gt_vel, dim=-1, index=last_idx.unsqueeze(-1)).squeeze(-1)

        label = torch.full_like(avg_accel, fill_value=2, dtype=torch.long)   # 默认匀速
        label[avg_accel < self.decel_label_thresh] = 1                       # 减速
        label[avg_accel > self.accel_label_thresh] = 3                       # 加速
        label[(avg_accel < self.brake_accel_thresh) | (end_vel < self.stop_vel_thresh)] = 0  # 急刹/停车,优先级最高
        return label

    def forward(self, pred_dict, gt_traj, gt_mask, agent_ctrs, valid_agent_mask=None):
        """
        pred_dict: AgentPredictor 输出 {'agent_trajs','agent_vel','behavior_logits'}
                   (若 MotionEstimation 额外返回 'deltas'，建议一并传入，见下方说明)
        gt_traj:    [bs, A, horizon, 2]  绝对坐标
        gt_mask:    [bs, A, horizon]
        agent_ctrs: [bs, A, 1, 2]        当前坐标，用于把 gt_traj 换算成 gt_delta 的起点对齐
        valid_agent_mask: [bs, A] 可选
        """
        agent_trajs = pred_dict['agent_trajs']
        agent_vel = pred_dict['agent_vel']
        behavior_logits = pred_dict['behavior_logits']
        bs, A, T, _ = agent_trajs.shape

        gt_delta, gt_vel, gt_accel = self._gt_kinematics(gt_traj, gt_mask)
        gt_behavior = self._generate_gt_behavior(gt_vel, gt_accel, gt_mask)

        decel_mask = (gt_accel < self.decel_accel_thresh).float()
        step_weight = 1.0 + self.decel_weight * decel_mask             # [bs, A, T]
        denom = (gt_mask * step_weight).sum().clamp(min=1.0)

        # --- delta 回归损失（核心：直接监督每一帧的位移，不被 cumsum 稀释） ---
        if 'deltas' in pred_dict:
            pred_delta = pred_dict['deltas']
        else:
            # 若 forward 未单独返回 deltas，从绝对轨迹反推（含拼接首帧）
            pred_delta = torch.zeros_like(agent_trajs)
            pred_delta[:, :, 0] = agent_trajs[:, :, 0] - agent_ctrs.squeeze(2)
            pred_delta[:, :, 1:] = agent_trajs[:, :, 1:] - agent_trajs[:, :, :-1]

        delta_err = F.smooth_l1_loss(pred_delta, gt_delta, reduction='none').sum(dim=-1)
        loss_delta = (delta_err * gt_mask * step_weight).sum() / denom

        # --- 绝对位置回归损失（反映累积误差下的真实部署精度） ---
        reg_err = F.smooth_l1_loss(agent_trajs, gt_traj, reduction='none').sum(dim=-1)
        loss_reg = (reg_err * gt_mask * step_weight).sum() / denom

        # --- 速度损失（标量、方向无关，天然适配任意行驶方向） ---
        vel_err = F.smooth_l1_loss(agent_vel, gt_vel, reduction='none')
        loss_vel = (vel_err * gt_mask * step_weight).sum() / denom

        # --- 行为分类损失 ---
        behavior_ce = F.cross_entropy(
            behavior_logits.reshape(bs * A, -1), gt_behavior.reshape(bs * A), reduction='none'
        ).reshape(bs, A)
        if valid_agent_mask is not None:
            va = valid_agent_mask.float()
            loss_behavior = (behavior_ce * va).sum() / va.sum().clamp(min=1.0)
        else:
            loss_behavior = behavior_ce.mean()

        total = (self.delta_weight * loss_delta
                 + self.reg_weight * loss_reg
                 + self.vel_weight * loss_vel
                 + self.behavior_weight * loss_behavior)

        return {
            'loss': total,
            'loss_delta': loss_delta.detach(),
            'loss_reg': loss_reg.detach(),
            'loss_vel': loss_vel.detach(),
            'loss_behavior': loss_behavior.detach(),
            'gt_behavior': gt_behavior.detach(),
        }