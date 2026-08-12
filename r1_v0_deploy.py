import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.mlp import Mlp


class AgentPredictor(nn.Module):
    """Deployment-friendly R1 predictor: behavior conditioning + parallel delta decoding."""
    def __init__(self, hidden_dim, future_len, num_behaviors=4, dt=0.08):
        super().__init__()
        self.behavior_estimator = BehaviorEstimation(hidden_dim, hidden_dim, num_behaviors)
        self.motion_estimator = MotionEstimation(hidden_dim, future_len, hidden_dim, dt=dt)

    def forward(self, agent_feats, agent_ctrs):
        behavior_logits, behavior_emb = self.behavior_estimator(agent_feats)
        agent_trajs, agent_vel, deltas = self.motion_estimator(agent_feats, agent_ctrs, behavior_emb)
        return {
            'agent_trajs': agent_trajs,
            'agent_vel': agent_vel,
            'behavior_logits': behavior_logits,
            'deltas': deltas,
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
    Parallel temporal decoder for vehicle deployment.

    Replaces horizon sequential GRUCell calls with one batched tensor path:
      feature -> FiLM behavior conditioning -> learned time queries -> gated MLP -> all deltas

    Output semantics remain delta -> cumsum absolute trajectory, and velocity is ||delta|| / dt.
    This graph has no Python autoregressive loop and is friendly to ONNX/TensorRT.
    """
    def __init__(self, in_channels, horizon=30, hidden_dim=64, use_conv=False, dt=0.08):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.dt = float(dt)

        self.feat_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_dim * 2,
            out_features=hidden_dim,
            drop=0.0,
            use_conv=use_conv,
            act_layer=nn.ReLU,
        )
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)

        # Learned per-future-step embeddings provide explicit temporal identity without recurrence.
        self.time_embed = nn.Parameter(torch.randn(horizon, hidden_dim) * 0.02)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # A multiplicative gate lets behavior-conditioned context modulate every future step.
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, agent_feats, agent_ctrs, behavior_emb):
        feat = self.feat_proj(agent_feats)
        gamma, beta = self.film(behavior_emb).chunk(2, dim=-1)
        cond_feat = feat * (1.0 + gamma) + beta                 # [B, A, H]

        context = self.context_proj(cond_feat).unsqueeze(2)     # [B, A, 1, H]
        gate = torch.sigmoid(self.gate_proj(cond_feat)).unsqueeze(2)
        time = self.time_proj(self.time_embed).view(1, 1, self.horizon, self.hidden_dim)

        temporal_feat = torch.tanh(context + time) * (1.0 + gate)
        deltas = self.delta_head(temporal_feat)                  # [B, A, T, 2]
        agent_trajs = agent_ctrs + torch.cumsum(deltas, dim=2)
        agent_vel = torch.linalg.vector_norm(deltas, dim=-1) / self.dt
        return agent_trajs, agent_vel, deltas


class AgentLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.dt = config.get('dt', 0.08)
        self.delta_weight = config.get('delta_weight', 1.0)
        self.reg_weight = config.get('reg_weight', 1.0)
        self.vel_weight = config.get('vel_weight', 1.0)
        self.behavior_weight = config.get('behavior_weight', 1.0)
        self.decel_weight = config.get('decel_weight', 2.0)
        self.decel_accel_thresh = config.get('decel_accel_thresh', -1.0)
        self.brake_accel_thresh = config.get('brake_accel_thresh', -3.0)
        self.decel_label_thresh = config.get('decel_label_thresh', -0.5)
        self.accel_label_thresh = config.get('accel_label_thresh', 0.5)
        self.stop_vel_thresh = config.get('stop_vel_thresh', 1.0)

    def _gt_kinematics(self, gt_traj, agent_ctrs):
        gt_delta = torch.zeros_like(gt_traj)
        gt_delta[:, :, 0] = gt_traj[:, :, 0] - agent_ctrs.squeeze(2)
        gt_delta[:, :, 1:] = gt_traj[:, :, 1:] - gt_traj[:, :, :-1]
        gt_vel = torch.linalg.vector_norm(gt_delta, dim=-1) / self.dt
        gt_accel = torch.zeros_like(gt_vel)
        gt_accel[:, :, 1:] = (gt_vel[:, :, 1:] - gt_vel[:, :, :-1]) / self.dt
        return gt_delta, gt_vel, gt_accel

    def _generate_gt_behavior(self, gt_vel, gt_accel, gt_mask):
        valid_len = gt_mask.sum(dim=-1).clamp(min=1.0)
        avg_accel = (gt_accel * gt_mask).sum(dim=-1) / valid_len
        last_idx = (valid_len - 1).long()
        end_vel = torch.gather(gt_vel, -1, last_idx.unsqueeze(-1)).squeeze(-1)
        label = torch.full_like(avg_accel, 2, dtype=torch.long)
        label[avg_accel < self.decel_label_thresh] = 1
        label[avg_accel > self.accel_label_thresh] = 3
        label[(avg_accel < self.brake_accel_thresh) | (end_vel < self.stop_vel_thresh)] = 0
        return label

    def forward(self, pred_dict, gt_traj, gt_mask, agent_ctrs, valid_agent_mask=None):
        agent_trajs = pred_dict['agent_trajs']
        agent_vel = pred_dict['agent_vel']
        behavior_logits = pred_dict['behavior_logits']
        pred_delta = pred_dict['deltas']
        bs, A = agent_trajs.shape[:2]

        gt_delta, gt_vel, gt_accel = self._gt_kinematics(gt_traj, agent_ctrs)
        gt_behavior = self._generate_gt_behavior(gt_vel, gt_accel, gt_mask)
        step_weight = 1.0 + self.decel_weight * (gt_accel < self.decel_accel_thresh).float()
        denom = (gt_mask * step_weight).sum().clamp(min=1.0)

        delta_err = F.smooth_l1_loss(pred_delta, gt_delta, reduction='none').sum(dim=-1)
        loss_delta = (delta_err * gt_mask * step_weight).sum() / denom
        reg_err = F.smooth_l1_loss(agent_trajs, gt_traj, reduction='none').sum(dim=-1)
        loss_reg = (reg_err * gt_mask * step_weight).sum() / denom
        vel_err = F.smooth_l1_loss(agent_vel, gt_vel, reduction='none')
        loss_vel = (vel_err * gt_mask * step_weight).sum() / denom

        behavior_ce = F.cross_entropy(
            behavior_logits.reshape(bs * A, -1), gt_behavior.reshape(bs * A), reduction='none'
        ).reshape(bs, A)
        if valid_agent_mask is not None:
            va = valid_agent_mask.float()
            loss_behavior = (behavior_ce * va).sum() / va.sum().clamp(min=1.0)
        else:
            loss_behavior = behavior_ce.mean()

        total = (self.delta_weight * loss_delta + self.reg_weight * loss_reg +
                 self.vel_weight * loss_vel + self.behavior_weight * loss_behavior)
        return {
            'loss': total,
            'loss_delta': loss_delta.detach(),
            'loss_reg': loss_reg.detach(),
            'loss_vel': loss_vel.detach(),
            'loss_behavior': loss_behavior.detach(),
            'gt_behavior': gt_behavior.detach(),
        }
