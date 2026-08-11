class AgentMotionLoss(nn.Module):
    """
    MTR-style Multi-modal Motion Loss

    pred_traj:
        [B,N,K,T,2]

    pred_score:
        [B,N,K]

    gt_traj:
        [B,N,T,2]

    gt_mask:
        [B,N,T]

    agent_mask:
        [B,N]
    """

    def __init__(
        self,
        num_modes=6,
        future_steps=75,

        cls_weight=1.0,
        reg_weight=1.0,
        fde_weight=1.0,
    ):
        super().__init__()

        self.num_modes = num_modes
        self.future_steps = future_steps

        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.fde_weight = fde_weight

    def forward(
        self,
        pred_traj,
        pred_score,
        gt_traj,
        gt_mask,
        agent_mask=None,
    ):
        """
        Args:
            pred_traj:
                [B,N,K,T,2]

            pred_score:
                [B,N,K]

            gt_traj:
                [B,N,T,2]

            gt_mask:
                [B,N,T]

            agent_mask:
                [B,N]
        """

        B, N, K, T, _ = pred_traj.shape

        assert K == self.num_modes
        assert T == self.future_steps

        # ==================================================
        # 1. Agent Mask
        # ==================================================

        if agent_mask is None:

            agent_mask = (
                gt_mask.sum(dim=-1) > 0
            )

        agent_mask = agent_mask.float()

        # ==================================================
        # 2. Calculate point-wise L2 distance
        # ==================================================

        gt = gt_traj.unsqueeze(2)

        # [B,N,1,T,2]

        diff = (
            pred_traj - gt
        )

        dist = torch.norm(
            diff,
            dim=-1,
        )

        # [B,N,K,T]

        # ==================================================
        # 3. Apply GT mask
        # ==================================================

        mask = gt_mask.unsqueeze(2).float()

        # [B,N,1,T]

        valid_num = mask.sum(
            dim=-1
        ).clamp_min(1.0)

        # ==================================================
        # 4. ADE
        # ==================================================

        ade = (
            dist * mask
        ).sum(dim=-1)

        ade = ade / valid_num

        # [B,N,K]

        # ==================================================
        # 5. FDE
        # ==================================================

        # 找到最后一个有效GT frame

        valid_length = (
            gt_mask.float().sum(
                dim=-1
            )
        )

        last_index = (
            valid_length.long() - 1
        ).clamp(
            min=0,
            max=T - 1,
        )

        batch_idx = torch.arange(
            B,
            device=pred_traj.device,
        ).view(B, 1)

        agent_idx = torch.arange(
            N,
            device=pred_traj.device,
        ).view(1, N)

        # GT final point
        gt_final = gt_traj[
            batch_idx,
            agent_idx,
            last_index,
        ]

        # [B,N,2]

        gt_final = gt_final.unsqueeze(2)

        # [B,N,1,2]

        # predicted final point
        pred_final = pred_traj[
            batch_idx,
            agent_idx,
            :,
            last_index,
        ]

        # [B,N,K,2]

        fde = torch.norm(
            pred_final - gt_final,
            dim=-1,
        )

        # [B,N,K]

        # ==================================================
        # 6. Best Mode Assignment
        # ==================================================
        #
        # 这里采用：
        #
        # ADE + FDE
        #
        # 找距离GT最近的mode
        #

        mode_error = (
            ade + fde
        )

        best_mode = torch.argmin(
            mode_error,
            dim=-1,
        )

        # [B,N]

        # ==================================================
        # 7. Classification Loss
        # ==================================================

        cls_loss = F.cross_entropy(
            pred_score.reshape(
                B * N,
                K,
            ),
            best_mode.reshape(
                B * N,
            ),
            reduction="none",
        )

        cls_loss = cls_loss.reshape(
            B,
            N,
        )

        cls_loss = (
            cls_loss * agent_mask
        ).sum()

        cls_loss = cls_loss / (
            agent_mask.sum().clamp_min(1.0)
        )

        # ==================================================
        # 8. Best Trajectory
        # ==================================================

        best_traj = pred_traj[
            batch_idx,
            agent_idx,
            best_mode,
        ]

        # [B,N,T,2]

        # ==================================================
        # 9. Regression Loss
        # ==================================================

        reg_error = torch.abs(
            best_traj - gt_traj
        )

        reg_error = (
            reg_error *
            gt_mask.unsqueeze(-1).float()
        )

        reg_loss = reg_error.sum()

        reg_loss = reg_loss / (
            gt_mask.sum().clamp_min(1.0)
            * 2.0
        )

        # ==================================================
        # 10. Best Mode FDE Loss
        # ==================================================

        best_fde = fde[
            batch_idx,
            agent_idx,
            best_mode,
        ]

        fde_loss = (
            best_fde * agent_mask
        ).sum()

        fde_loss = fde_loss / (
            agent_mask.sum().clamp_min(1.0)
        )

        # ==================================================
        # 11. Total Loss
        # ==================================================

        total_loss = (
            self.cls_weight * cls_loss
            +
            self.reg_weight * reg_loss
            +
            self.fde_weight * fde_loss
        )

        return {
            "loss": total_loss,

            "cls_loss": cls_loss.detach(),

            "reg_loss": reg_loss.detach(),

            "fde_loss": fde_loss.detach(),

            "best_mode": best_mode.detach(),

            "ade": ade.detach(),

            "fde": fde.detach(),
        }
