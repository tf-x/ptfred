# =============================================================================
# Merged version based on r1_v0_deploy_asym_decel(2).py
#
# Added conservative-margin supervision for deceleration / hard-brake:
#   1) explicit target shortening margin
#   2) gradually increasing per-frame margin after deceleration starts
#   3) stronger endpoint conservative loss
#   4) endpoint overshoot / undershoot diagnostics
#
# AgentPredictor / MotionEstimation inference graph is unchanged.
# All additional logic is training-loss-only.
# =============================================================================

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
    """
    Conservative deceleration loss with an explicit shortening target.

    Difference vs. the previous asymmetric loss:
      previous: GT is still the optimum; shorter predictions are merely penalized less.
      this version: during deceleration, the target is shifted backward by a gradually
      increasing conservative margin, so the optimum is intentionally shorter than GT.

    Inference graph is unchanged. All added logic is training-loss-only.
    """
    def __init__(self, config=None):
        super().__init__()
        config = config or {}

        self.dt = config.get('dt', 0.08)

        # Base symmetric supervision.
        self.delta_weight = config.get('delta_weight', 1.0)
        self.reg_weight = config.get('reg_weight', 1.0)
        self.vel_weight = config.get('vel_weight', 1.0)
        self.behavior_weight = config.get('behavior_weight', 1.0)

        # Symmetric GT fitting is kept weak in a deceleration phase so it does not
        # fight strongly against the deliberately-short conservative target.
        self.decel_symmetric_scale = config.get('decel_symmetric_scale', 0.15)

        # Frame-wise conservative progress loss.
        self.progress_asym_weight = config.get('progress_asym_weight', 1.0)
        self.progress_over_weight = config.get('progress_over_weight', 4.0)
        self.progress_under_weight = config.get('progress_under_weight', 0.5)

        # Endpoint conservative loss is intentionally stronger because a frame-average
        # loss can otherwise dilute a large final overshoot across the whole horizon.
        self.endpoint_asym_weight = config.get('endpoint_asym_weight', 2.0)
        self.endpoint_over_weight = config.get('endpoint_over_weight', 6.0)
        self.endpoint_under_weight = config.get('endpoint_under_weight', 0.5)

        # Conservative shortening target at the end of the horizon.
        # Default: ~6% of GT path length, clamped to [0.5m, 2.0m].
        # Example: GT path length 25m -> target about 1.5m shorter.
        self.conservative_margin_ratio = config.get('conservative_margin_ratio', 0.06)
        self.conservative_margin_min_m = config.get('conservative_margin_min_m', 0.5)
        self.conservative_margin_max_m = config.get('conservative_margin_max_m', 2.0)
        self.margin_power = config.get('margin_power', 1.5)

        # Velocity remains asymmetric around GT. The explicit trajectory margin above
        # is what creates the shortening bias; this term mainly prevents high-speed overshoot.
        self.vel_asym_weight = config.get('vel_asym_weight', 1.0)
        self.vel_over_weight = config.get('vel_over_weight', 3.0)
        self.vel_under_weight = config.get('vel_under_weight', 0.5)

        self.conservative_accel_thresh = config.get('conservative_accel_thresh', -0.5)

        self.brake_accel_thresh = config.get('brake_accel_thresh', -3.0)
        self.decel_label_thresh = config.get('decel_label_thresh', -0.5)
        self.accel_label_thresh = config.get('accel_label_thresh', 0.5)
        self.stop_vel_thresh = config.get('stop_vel_thresh', 1.0)

        self.tangent_eps = config.get('tangent_eps', 1e-4)

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
        last_idx = (valid_len - 1).clamp(min=0).long()
        end_vel = torch.gather(gt_vel, -1, last_idx.unsqueeze(-1)).squeeze(-1)

        label = torch.full_like(avg_accel, 2, dtype=torch.long)
        label[avg_accel < self.decel_label_thresh] = 1
        label[avg_accel > self.accel_label_thresh] = 3
        label[(avg_accel < self.brake_accel_thresh) |
              (end_vel < self.stop_vel_thresh)] = 0
        return label

    def _build_decel_phase_mask(self, gt_behavior, gt_accel, gt_mask):
        decel_agent = ((gt_behavior == 0) | (gt_behavior == 1)).unsqueeze(-1)
        decel_event = gt_accel < self.conservative_accel_thresh
        after_first_decel = torch.cumsum(decel_event.to(torch.int32), dim=-1) > 0
        return (decel_agent & after_first_decel).float() * gt_mask.float()

    def _gt_tangent(self, gt_delta):
        speed_norm = torch.linalg.vector_norm(gt_delta, dim=-1, keepdim=True)
        valid_dir = speed_norm > self.tangent_eps
        tangent = gt_delta / speed_norm.clamp(min=self.tangent_eps)

        tangent = tangent.clone()
        valid = valid_dir.clone()
        for t in range(1, tangent.shape[2]):
            use_prev = ~valid[:, :, t]
            tangent[:, :, t] = torch.where(
                use_prev.expand_as(tangent[:, :, t]),
                tangent[:, :, t - 1],
                tangent[:, :, t],
            )
            valid[:, :, t] = valid[:, :, t] | valid[:, :, t - 1]
        return tangent, valid.squeeze(-1).float()

    def _build_margin_profile(self, gt_delta, gt_mask, decel_mask):
        """
        Returns:
          margin_profile [B,A,T]: desired longitudinal shortening per future frame.
          margin_end     [B,A]:   desired shortening at final valid frame.

        Margin ramps from ~0 at the first deceleration frame to margin_end at horizon end.
        """
        gt_step = torch.linalg.vector_norm(gt_delta, dim=-1) * gt_mask
        gt_path_len = gt_step.sum(dim=-1)  # [B,A]

        margin_end = (gt_path_len * self.conservative_margin_ratio).clamp(
            min=self.conservative_margin_min_m,
            max=self.conservative_margin_max_m,
        )

        # Only agents with an active decel phase should receive a non-zero margin.
        has_decel = (decel_mask.sum(dim=-1) > 0).float()
        margin_end = margin_end * has_decel

        phase_index = torch.cumsum(decel_mask, dim=-1)
        phase_len = decel_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        phase = (phase_index / phase_len).clamp(0.0, 1.0)
        phase = phase.pow(self.margin_power)
        margin_profile = margin_end.unsqueeze(-1) * phase * decel_mask
        return margin_profile, margin_end

    @staticmethod
    def _smooth_l1_to_zero(x):
        return F.smooth_l1_loss(x, torch.zeros_like(x), reduction='none')

    def _conservative_progress_loss(
        self,
        agent_trajs,
        gt_traj,
        gt_delta,
        decel_mask,
        margin_profile,
    ):
        tangent, tangent_valid = self._gt_tangent(gt_delta)

        # Signed progress error relative to GT:
        # >0: farther than GT; <0: shorter than GT.
        pos_error = agent_trajs - gt_traj
        progress_error = (pos_error * tangent).sum(dim=-1)

        # Desired error is -margin_profile, not zero.
        # residual >0: prediction is longer than our conservative target.
        # residual <0: prediction is even shorter than the conservative target.
        residual = progress_error + margin_profile
        too_long = torch.relu(residual)
        too_short = torch.relu(-residual)

        err = (
            self.progress_over_weight * self._smooth_l1_to_zero(too_long)
            + self.progress_under_weight * self._smooth_l1_to_zero(too_short)
        )

        # Slightly emphasize later future frames, because endpoint length matters most here.
        T = agent_trajs.shape[2]
        time_w = torch.linspace(0.5, 1.5, T, device=agent_trajs.device, dtype=agent_trajs.dtype)
        time_w = time_w.view(1, 1, T)

        mask = decel_mask * tangent_valid
        denom = (mask * time_w).sum().clamp(min=1.0)
        loss = (err * mask * time_w).sum() / denom

        # Diagnostics relative to the original GT (business metric) and shifted target.
        gt_over = torch.relu(progress_error)
        gt_under = torch.relu(-progress_error)
        active_count = mask.sum().clamp(min=1.0)
        gt_over_count = ((gt_over > 0) & (mask > 0)).float().sum()
        gt_under_count = ((gt_under > 0) & (mask > 0)).float().sum()
        target_over_count = ((residual > 0) & (mask > 0)).float().sum()

        diagnostics = {
            'decel_overshoot_rate': gt_over_count / active_count,
            'decel_undershoot_rate': gt_under_count / active_count,
            'decel_mean_overshoot_m': (gt_over * mask).sum() / gt_over_count.clamp(min=1.0),
            'decel_mean_undershoot_m': (gt_under * mask).sum() / gt_under_count.clamp(min=1.0),
            'decel_conservative_target_over_rate': target_over_count / active_count,
        }
        return loss, progress_error, diagnostics

    def _endpoint_conservative_loss(
        self,
        progress_error,
        margin_end,
        gt_mask,
        decel_mask,
    ):
        """Strong final-horizon loss so endpoint overshoot is not diluted by frame averaging."""
        valid_len = gt_mask.sum(dim=-1).clamp(min=1.0)
        last_idx = (valid_len - 1).long()

        end_progress_error = torch.gather(
            progress_error, -1, last_idx.unsqueeze(-1)
        ).squeeze(-1)

        has_decel = decel_mask.sum(dim=-1) > 0
        residual = end_progress_error + margin_end  # target = -margin_end

        too_long = torch.relu(residual)
        too_short = torch.relu(-residual)
        err = (
            self.endpoint_over_weight * self._smooth_l1_to_zero(too_long)
            + self.endpoint_under_weight * self._smooth_l1_to_zero(too_short)
        )

        mask = has_decel.float()
        denom = mask.sum().clamp(min=1.0)
        loss = (err * mask).sum() / denom

        # Business-facing endpoint diagnostics relative to GT.
        gt_over = torch.relu(end_progress_error)
        gt_under = torch.relu(-end_progress_error)
        over_count = ((gt_over > 0) & has_decel).float().sum()
        under_count = ((gt_under > 0) & has_decel).float().sum()

        diagnostics = {
            'decel_endpoint_progress_diff_m': (end_progress_error * mask).sum() / denom,
            'decel_endpoint_overshoot_rate': over_count / denom,
            'decel_endpoint_undershoot_rate': under_count / denom,
            'decel_endpoint_mean_overshoot_m': (gt_over * mask).sum() / over_count.clamp(min=1.0),
            'decel_endpoint_mean_undershoot_m': (gt_under * mask).sum() / under_count.clamp(min=1.0),
            'decel_target_margin_m': (margin_end * mask).sum() / denom,
        }
        return loss, diagnostics

    def _asymmetric_velocity_loss(self, agent_vel, gt_vel, decel_mask):
        vel_diff = agent_vel - gt_vel
        vel_over = torch.relu(vel_diff)
        vel_under = torch.relu(-vel_diff)

        err = (
            self.vel_over_weight * self._smooth_l1_to_zero(vel_over)
            + self.vel_under_weight * self._smooth_l1_to_zero(vel_under)
        )
        denom = decel_mask.sum().clamp(min=1.0)
        loss = (err * decel_mask).sum() / denom

        over_count = ((vel_over > 0) & (decel_mask > 0)).float().sum()
        active_count = decel_mask.sum().clamp(min=1.0)
        diagnostics = {
            'decel_speed_over_rate': over_count / active_count,
            'decel_mean_speed_over_mps': (vel_over * decel_mask).sum() / over_count.clamp(min=1.0),
        }
        return loss, diagnostics

    def forward(self, pred_dict, gt_traj, gt_mask, agent_ctrs, valid_agent_mask=None):
        agent_trajs = pred_dict['agent_trajs']
        agent_vel = pred_dict['agent_vel']
        behavior_logits = pred_dict['behavior_logits']
        pred_delta = pred_dict['deltas']
        bs, A = agent_trajs.shape[:2]

        gt_mask = gt_mask.float()
        if valid_agent_mask is not None:
            gt_mask = gt_mask * valid_agent_mask.float().unsqueeze(-1)

        gt_delta, gt_vel, gt_accel = self._gt_kinematics(gt_traj, agent_ctrs)
        gt_behavior = self._generate_gt_behavior(gt_vel, gt_accel, gt_mask)
        decel_mask = self._build_decel_phase_mask(gt_behavior, gt_accel, gt_mask)
        margin_profile, margin_end = self._build_margin_profile(gt_delta, gt_mask, decel_mask)

        # 1) Weak symmetric anchor in decel; full symmetric fitting elsewhere.
        symmetric_weight = gt_mask * (
            1.0 - decel_mask + self.decel_symmetric_scale * decel_mask
        )
        symmetric_denom = symmetric_weight.sum().clamp(min=1.0)

        delta_err = F.smooth_l1_loss(pred_delta, gt_delta, reduction='none').sum(dim=-1)
        loss_delta = (delta_err * symmetric_weight).sum() / symmetric_denom

        reg_err = F.smooth_l1_loss(agent_trajs, gt_traj, reduction='none').sum(dim=-1)
        loss_reg = (reg_err * symmetric_weight).sum() / symmetric_denom

        vel_err = F.smooth_l1_loss(agent_vel, gt_vel, reduction='none')
        loss_vel = (vel_err * symmetric_weight).sum() / symmetric_denom

        # 2) Explicit shifted conservative target during deceleration.
        loss_progress_asym, progress_error, progress_diag = self._conservative_progress_loss(
            agent_trajs, gt_traj, gt_delta, decel_mask, margin_profile
        )
        loss_endpoint_asym, endpoint_diag = self._endpoint_conservative_loss(
            progress_error, margin_end, gt_mask, decel_mask
        )
        loss_vel_asym, vel_diag = self._asymmetric_velocity_loss(
            agent_vel, gt_vel, decel_mask
        )

        # 3) Behavior classification.
        behavior_ce = F.cross_entropy(
            behavior_logits.reshape(bs * A, -1),
            gt_behavior.reshape(bs * A),
            reduction='none',
        ).reshape(bs, A)

        if valid_agent_mask is not None:
            va = valid_agent_mask.float()
            loss_behavior = (behavior_ce * va).sum() / va.sum().clamp(min=1.0)
        else:
            loss_behavior = behavior_ce.mean()

        total = (
            self.delta_weight * loss_delta
            + self.reg_weight * loss_reg
            + self.vel_weight * loss_vel
            + self.behavior_weight * loss_behavior
            + self.progress_asym_weight * loss_progress_asym
            + self.endpoint_asym_weight * loss_endpoint_asym
            + self.vel_asym_weight * loss_vel_asym
        )

        return {
            'loss': total,
            'loss_delta': loss_delta.detach(),
            'loss_reg': loss_reg.detach(),
            'loss_vel': loss_vel.detach(),
            'loss_behavior': loss_behavior.detach(),
            'loss_progress_asym': loss_progress_asym.detach(),
            'loss_endpoint_asym': loss_endpoint_asym.detach(),
            'loss_vel_asym': loss_vel_asym.detach(),
            'gt_behavior': gt_behavior.detach(),
            'decel_phase_ratio': (decel_mask.sum() / gt_mask.sum().clamp(min=1.0)).detach(),
            **{k: v.detach() for k, v in progress_diag.items()},
            **{k: v.detach() for k, v in endpoint_diag.items()},
            **{k: v.detach() for k, v in vel_diag.items()},
        }
