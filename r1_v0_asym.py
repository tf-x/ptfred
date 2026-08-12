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
    R1 training loss with conservative asymmetric supervision for deceleration.

    Goal in a deceleration / hard-brake phase:
      - prediction farther than GT along the GT travel direction (overshoot): strong penalty
      - prediction shorter than GT (undershoot): weak penalty

    Non-deceleration frames keep the ordinary symmetric regression objective.
    The inference graph in AgentPredictor / MotionEstimation is unchanged.
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

        # During a deceleration phase, symmetric regression is intentionally weakened.
        # 1.0 = original symmetric supervision; 0.0 = no symmetric supervision there.
        # A small non-zero value keeps the trajectory close to GT without forcing exact length.
        self.decel_symmetric_scale = config.get('decel_symmetric_scale', 0.25)

        # Conservative asymmetric terms.
        self.progress_asym_weight = config.get('progress_asym_weight', 1.0)
        self.progress_over_weight = config.get('progress_over_weight', 4.0)
        self.progress_under_weight = config.get('progress_under_weight', 0.5)

        self.vel_asym_weight = config.get('vel_asym_weight', 1.0)
        self.vel_over_weight = config.get('vel_over_weight', 3.0)
        self.vel_under_weight = config.get('vel_under_weight', 0.5)

        # Deceleration phase starts at the first frame whose GT acceleration crosses this threshold,
        # and remains active to the end of the valid horizon. This also covers the low-speed / stopped
        # tail after braking has finished.
        self.conservative_accel_thresh = config.get('conservative_accel_thresh', -0.5)

        # Behavior-label thresholds kept from the original implementation.
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

        label = torch.full_like(avg_accel, 2, dtype=torch.long)  # constant
        label[avg_accel < self.decel_label_thresh] = 1          # decelerate
        label[avg_accel > self.accel_label_thresh] = 3          # accelerate
        label[(avg_accel < self.brake_accel_thresh) |
              (end_vel < self.stop_vel_thresh)] = 0             # hard brake / stop
        return label

    def _build_decel_phase_mask(self, gt_behavior, gt_accel, gt_mask):
        """
        [B,A,T] mask. Only agents labeled decel/hard-brake can enter conservative mode.
        Once a real deceleration frame is observed, the mask stays active until the end of
        the valid future. This prevents the post-brake constant-speed/stopped tail from being
        pulled back to an exactly-GT trajectory length by the symmetric loss.
        """
        decel_agent = ((gt_behavior == 0) | (gt_behavior == 1)).unsqueeze(-1)
        decel_event = gt_accel < self.conservative_accel_thresh
        after_first_decel = torch.cumsum(decel_event.to(torch.int32), dim=-1) > 0
        return (decel_agent & after_first_decel).float() * gt_mask.float()

    def _gt_tangent(self, gt_delta):
        """
        Per-frame GT travel direction [B,A,T,2]. For a stopped tail (delta ~= 0), carry
        forward the last valid direction so progress remains well-defined after braking.
        The small loop is training-loss-only and is not part of the deployment inference graph.
        """
        speed_norm = torch.linalg.vector_norm(gt_delta, dim=-1, keepdim=True)
        valid_dir = speed_norm > self.tangent_eps
        tangent = gt_delta / speed_norm.clamp(min=self.tangent_eps)

        # Carry the last valid tangent through zero-motion frames (e.g. stopped tail).
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

    @staticmethod
    def _smooth_l1_to_zero(x):
        return F.smooth_l1_loss(x, torch.zeros_like(x), reduction='none')

    def _asymmetric_progress_loss(
        self,
        agent_trajs,
        gt_traj,
        gt_delta,
        decel_mask,
    ):
        tangent, tangent_valid = self._gt_tangent(gt_delta)

        # Signed longitudinal position error in the local GT travel direction:
        #   > 0 : prediction is farther than GT  -> overshoot
        #   < 0 : prediction is shorter than GT -> undershoot
        pos_error = agent_trajs - gt_traj
        progress_error = (pos_error * tangent).sum(dim=-1)

        overshoot = torch.relu(progress_error)
        undershoot = torch.relu(-progress_error)

        err = (
            self.progress_over_weight * self._smooth_l1_to_zero(overshoot)
            + self.progress_under_weight * self._smooth_l1_to_zero(undershoot)
        )

        mask = decel_mask * tangent_valid
        denom = mask.sum().clamp(min=1.0)
        loss = (err * mask).sum() / denom

        # Diagnostics use distances rather than SmoothL1 values.
        overshoot_count = ((overshoot > 0) & (mask > 0)).float().sum()
        undershoot_count = ((undershoot > 0) & (mask > 0)).float().sum()
        active_count = mask.sum().clamp(min=1.0)

        diagnostics = {
            'decel_overshoot_rate': overshoot_count / active_count,
            'decel_undershoot_rate': undershoot_count / active_count,
            'decel_mean_overshoot_m': (overshoot * mask).sum() / overshoot_count.clamp(min=1.0),
            'decel_mean_undershoot_m': (undershoot * mask).sum() / undershoot_count.clamp(min=1.0),
        }
        return loss, diagnostics

    def _asymmetric_velocity_loss(self, agent_vel, gt_vel, decel_mask):
        # Positive error means predicted speed is too high -> trajectory tends to be too long.
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

        # ------------------------------------------------------------------
        # 1) Base symmetric supervision.
        #    Full weight outside deceleration; intentionally weaker after braking starts.
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 2) Conservative deceleration supervision.
        #    Shorter/slower than GT is allowed with a small penalty;
        #    farther/faster than GT is penalized much more strongly.
        # ------------------------------------------------------------------
        loss_progress_asym, progress_diag = self._asymmetric_progress_loss(
            agent_trajs, gt_traj, gt_delta, decel_mask
        )
        loss_vel_asym, vel_diag = self._asymmetric_velocity_loss(
            agent_vel, gt_vel, decel_mask
        )

        # ------------------------------------------------------------------
        # 3) Behavior classification.
        # ------------------------------------------------------------------
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
            + self.vel_asym_weight * loss_vel_asym
        )

        return {
            'loss': total,
            'loss_delta': loss_delta.detach(),
            'loss_reg': loss_reg.detach(),
            'loss_vel': loss_vel.detach(),
            'loss_behavior': loss_behavior.detach(),
            'loss_progress_asym': loss_progress_asym.detach(),
            'loss_vel_asym': loss_vel_asym.detach(),
            'gt_behavior': gt_behavior.detach(),
            'decel_phase_ratio': (decel_mask.sum() / gt_mask.sum().clamp(min=1.0)).detach(),
            **{k: v.detach() for k, v in progress_diag.items()},
            **{k: v.detach() for k, v in vel_diag.items()},
        }
