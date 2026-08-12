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
        """
        agent_feats: [bs, num_agents, hidden_dim]  agent (x,y) 的高维特征表示（唯一输入）
        agent_ctrs:  [bs, num_agents, 1, 2]        当前坐标，作为轨迹积分基准点
        """
        behavior_logits, behavior_emb = self.behavior_estimator(agent_feats)
        agent_trajs, agent_vel = self.motion_estimator(agent_feats, agent_ctrs, behavior_emb)
        return {
            'agent_trajs': agent_trajs,          # [bs, A, horizon, 2]
            'agent_vel': agent_vel,              # [bs, A, horizon]
            'behavior_logits': behavior_logits,  # [bs, A, num_behaviors]
        }


class BehaviorEstimation(nn.Module):
    """
    纵向行为分类分支：0急刹 1减速 2匀速 3加速
    训练用 gt_behavior 伪标签监督（由 GT 未来速度变化生成，见 AgentLoss）
    """
    def __init__(self, in_channels, hidden_dim, num_behaviors=4):
        super().__init__()
        self.num_behaviors = num_behaviors
        self.cls_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_behaviors),
        )
        self.behavior_embed = nn.Embedding(num_behaviors, hidden_dim)

    def forward(self, agent_feats):
        behavior_logits = self.cls_head(agent_feats)                  # [bs, A, num_behaviors]
        behavior_probs = torch.softmax(behavior_logits, dim=-1)
        behavior_emb = behavior_probs @ self.behavior_embed.weight     # 可导软嵌入 [bs, A, hidden]
        return behavior_logits, behavior_emb


class MotionEstimation(nn.Module):
    def __init__(self,
                 in_channels,
                 horizon=30,
                 hidden_dim=64,
                 use_conv=False):
        super().__init__()
        self.in_channels = in_channels
        self.horizon = horizon           # 修复: 原代码 self.horizen = self.horizen 会报错(未定义)
        self.hidden_dim = hidden_dim

        self.feat_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_dim * 2,
            out_features=hidden_dim,
            drop=0.0,
            use_conv=use_conv,
            act_layer=nn.ReLU,
        )

        # FiLM: behavior_emb -> (gamma, beta)，对轨迹特征做仿射调制
        # 让"减速/急刹"这个条件真正参与轨迹生成，而不是解码完再去对齐
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)

        self.traj_pred = nn.Sequential(
            Mlp(
                in_features=hidden_dim,
                hidden_features=hidden_dim // 2,
                out_features=hidden_dim,
                drop=0.0,
                use_conv=use_conv,
                act_layer=nn.ReLU,
            ),
            nn.Linear(hidden_dim, horizon * 2),
        )

        # 速度轮廓辅助头，与轨迹共享 FiLM 后特征，显式监督减速曲线
        self.vel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, horizon),
            nn.Softplus(),   # 速度非负
        )

    def forward(self, agent_feats, agent_ctrs, behavior_emb):
        bs, A = agent_feats.shape[:2]

        feat = self.feat_proj(agent_feats)                       # [bs, A, hidden]

        gamma, beta = self.film(behavior_emb).chunk(2, dim=-1)
        feat = feat * (1.0 + gamma) + beta                        # 行为条件注入

        offsets = self.traj_pred(feat).view(bs, A, self.horizon, 2)
        agent_trajs = offsets + agent_ctrs                        # 相对位移 + 当前坐标

        agent_vel = self.vel_head(feat)                            # [bs, A, horizon]
        return agent_trajs, agent_vel


class AgentLoss(nn.Module):
    """
    - gt_behavior 由 GT 未来速度变化自动生成，无需人工标注
    - 减速片段位置误差加权
    - 显式速度监督 + 轨迹-速度一致性约束（防止 vel_head 与 traj 位移"各说各话"）
    """
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.dt = config.get('dt', 0.1)
        self.reg_weight = config.get('reg_weight', 1.0)
        self.vel_weight = config.get('vel_weight', 1.0)
        self.consist_weight = config.get('consist_weight', 0.5)
        self.behavior_weight = config.get('behavior_weight', 1.0)
        self.decel_weight = config.get('decel_weight', 2.0)
        self.decel_accel_thresh = config.get('decel_accel_thresh', -1.0)
        self.brake_accel_thresh = config.get('brake_accel_thresh', -3.0)
        self.decel_label_thresh = config.get('decel_label_thresh', -0.5)
        self.accel_label_thresh = config.get('accel_label_thresh', 0.5)
        self.stop_vel_thresh = config.get('stop_vel_thresh', 1.0)

    def _gt_kinematics(self, gt_traj, gt_mask):
        gt_vel = torch.zeros_like(gt_traj[..., 0])
        gt_vel[:, :, 1:] = torch.norm(gt_traj[:, :, 1:] - gt_traj[:, :, :-1], dim=-1) / self.dt
        gt_vel[:, :, 0] = gt_vel[:, :, 1]
        gt_accel = torch.zeros_like(gt_vel)
        gt_accel[:, :, 1:] = (gt_vel[:, :, 1:] - gt_vel[:, :, :-1]) / self.dt
        return gt_vel, gt_accel

    def _generate_gt_behavior(self, gt_vel, gt_accel, gt_mask):
        valid_len = gt_mask.sum(dim=-1).clamp(min=1.0)
        avg_accel = (gt_accel * gt_mask).sum(dim=-1) / valid_len

        last_idx = (valid_len - 1).clamp(min=0).long()
        end_vel = torch.gather(gt_vel, dim=-1, index=last_idx.unsqueeze(-1)).squeeze(-1)

        label = torch.full_like(avg_accel, fill_value=2, dtype=torch.long)   # 默认匀速
        label[avg_accel < self.decel_label_thresh] = 1                       # 减速
        label[avg_accel > self.accel_label_thresh] = 3                       # 加速
        label[(avg_accel < self.brake_accel_thresh) | (end_vel < self.stop_vel_thresh)] = 0  # 急刹/停车
        return label

    def forward(self, pred_dict, gt_traj, gt_mask, valid_agent_mask=None):
        """
        pred_dict: AgentPredictor 输出 {'agent_trajs','agent_vel','behavior_logits'}
        gt_traj:   [bs, A, horizon, 2]
        gt_mask:   [bs, A, horizon]
        valid_agent_mask: [bs, A] 可选
        """
        agent_trajs, agent_vel, behavior_logits = (
            pred_dict['agent_trajs'], pred_dict['agent_vel'], pred_dict['behavior_logits']
        )
        bs, A, T, _ = agent_trajs.shape

        gt_vel, gt_accel = self._gt_kinematics(gt_traj, gt_mask)
        gt_behavior = self._generate_gt_behavior(gt_vel, gt_accel, gt_mask)

        decel_mask = (gt_accel < self.decel_accel_thresh).float()
        step_weight = 1.0 + self.decel_weight * decel_mask
        denom = (gt_mask * step_weight).sum().clamp(min=1.0)

        # 位置回归损失（减速片段加权）
        reg_err = F.smooth_l1_loss(agent_trajs, gt_traj, reduction='none').sum(dim=-1)
        loss_reg = (reg_err * gt_mask * step_weight).sum() / denom

        # 速度轮廓监督
        vel_err = F.smooth_l1_loss(agent_vel, gt_vel, reduction='none')
        loss_vel = (vel_err * gt_mask * step_weight).sum() / denom

        # 一致性约束：由预测轨迹反算的位移速度 应与 vel_head 输出接近
        pred_vel_from_traj = torch.zeros_like(agent_vel)
        pred_vel_from_traj[:, :, 1:] = torch.norm(
            agent_trajs[:, :, 1:] - agent_trajs[:, :, :-1], dim=-1
        ) / self.dt
        pred_vel_from_traj[:, :, 0] = pred_vel_from_traj[:, :, 1]
        consist_err = F.smooth_l1_loss(pred_vel_from_traj, agent_vel, reduction='none')
        loss_consist = (consist_err * gt_mask).sum() / gt_mask.sum().clamp(min=1.0)

        # 行为分类损失
        behavior_ce = F.cross_entropy(
            behavior_logits.reshape(bs * A, -1), gt_behavior.reshape(bs * A), reduction='none'
        ).reshape(bs, A)
        if valid_agent_mask is not None:
            va = valid_agent_mask.float()
            loss_behavior = (behavior_ce * va).sum() / va.sum().clamp(min=1.0)
        else:
            loss_behavior = behavior_ce.mean()

        total = (self.reg_weight * loss_reg
                 + self.vel_weight * loss_vel
                 + self.consist_weight * loss_consist
                 + self.behavior_weight * loss_behavior)

        return {
            'loss': total,
            'loss_reg': loss_reg.detach(),
            'loss_vel': loss_vel.detach(),
            'loss_consist': loss_consist.detach(),
            'loss_behavior': loss_behavior.detach(),
            'gt_behavior': gt_behavior.detach(),
        }
