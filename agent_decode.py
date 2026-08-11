import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        dropout=0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MotionDecoderLayer(nn.Module):
    """
    MTR-style Motion Decoder Layer

    query:
        [BN, K, C]

    agent_feat:
        [BN, 1, C]
    """

    def __init__(
        self,
        hidden_dim=256,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
    ):
        super().__init__()

        # --------------------------------------------------
        # Motion Query Self Attention
        # --------------------------------------------------
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --------------------------------------------------
        # Motion Query -> Agent Feature
        # --------------------------------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, agent_feat):

        # ==================================================
        # 1. Motion Query Self Attention
        # ==================================================

        q = self.norm1(query)

        self_attn_out, _ = self.self_attn(
            q,
            q,
            q,
        )

        query = query + self_attn_out

        # ==================================================
        # 2. Cross Attention
        # ==================================================

        q = self.norm2(query)

        cross_attn_out, _ = self.cross_attn(
            q,
            agent_feat,
            agent_feat,
        )

        query = query + cross_attn_out

        # ==================================================
        # 3. FFN
        # ==================================================

        query = query + self.ffn(
            self.norm3(query)
        )

        return query


class AgentMotionDecoder(nn.Module):
    """
    MTR-style Agent Motion Decoder

    Input:
        agent_feat:
            [B, N, 256]

    Motion intention points:
        [K, 2]

    Output:
        pred_traj:
            [B, N, K, 75, 2]

        pred_score:
            [B, N, K]
    """

    def __init__(
        self,
        input_dim=256,
        hidden_dim=256,
        num_modes=6,
        future_steps=75,
        num_heads=8,
        num_layers=3,
        ff_dim=1024,
        dropout=0.1,
        intention_points=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.future_steps = future_steps

        # ==================================================
        # 1. Agent Feature Projection
        # ==================================================

        self.agent_proj = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ==================================================
        # 2. Intention Point -> Motion Query
        # ==================================================
        #
        # MTR核心思想：
        #
        # cluster center (x, y)
        #       ↓
        #      MLP
        #       ↓
        # Motion Query
        #
        # 不人为限制 x > 0
        #

        if intention_points is None:

            # 如果没有传入聚类中心，
            # 使用随机初始化。
            #
            # 实际项目建议传入真实K-means结果。

            self.intention_points = nn.Parameter(
                torch.randn(
                    num_modes,
                    2,
                )
            )

        else:

            assert intention_points.shape == (
                num_modes,
                2,
            )

            self.register_buffer(
                "intention_points",
                torch.tensor(
                    intention_points,
                    dtype=torch.float32,
                ),
            )

        self.intention_encoder = nn.Sequential(
            nn.Linear(
                2,
                hidden_dim // 2,
            ),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim // 2,
                hidden_dim,
            ),
        )

        # ==================================================
        # 3. Agent Feature + Motion Query
        # ==================================================

        self.query_fusion = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ==================================================
        # 4. Motion Decoder
        # ==================================================

        self.decoder_layers = nn.ModuleList(
            [
                MotionDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # ==================================================
        # 5. Trajectory Prediction Head
        # ==================================================

        self.traj_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                future_steps * 2,
            ),
        )

        # ==================================================
        # 6. Confidence Head
        # ==================================================

        self.score_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim // 2,
                1,
            ),
        )

    def forward(self, agent_feat):

        """
        agent_feat:
            [B, N, C]

        return:
            pred_traj:
                [B, N, K, T, 2]

            pred_score:
                [B, N, K]
        """

        B, N, C = agent_feat.shape

        assert C == self.input_dim

        # ==================================================
        # 1. Agent Feature
        # ==================================================

        agent_feat = self.agent_proj(
            agent_feat
        )

        # [B,N,256]

        # ==================================================
        # 2. Encode intention points
        # ==================================================

        motion_query = self.intention_encoder(
            self.intention_points
        )

        # [K,256]

        # ==================================================
        # 3. Expand to every Agent
        # ==================================================

        motion_query = motion_query.unsqueeze(0)

        motion_query = motion_query.unsqueeze(0)

        # [1,1,K,256]

        motion_query = motion_query.expand(
            B,
            N,
            self.num_modes,
            self.hidden_dim,
        )

        # [B,N,K,256]

        # ==================================================
        # 4. Agent-conditioned Query
        # ==================================================

        agent_context = agent_feat.unsqueeze(2)

        agent_context = agent_context.expand(
            B,
            N,
            self.num_modes,
            self.hidden_dim,
        )

        query = torch.cat(
            [
                motion_query,
                agent_context,
            ],
            dim=-1,
        )

        query = self.query_fusion(
            query
        )

        # [B,N,K,256]

        # ==================================================
        # 5. Flatten B,N
        # ==================================================

        query = query.reshape(
            B * N,
            self.num_modes,
            self.hidden_dim,
        )

        agent_context = agent_feat.reshape(
            B * N,
            1,
            self.hidden_dim,
        )

        # ==================================================
        # 6. Motion Decoder
        # ==================================================

        for layer in self.decoder_layers:

            query = layer(
                query=query,
                agent_feat=agent_context,
            )

        # ==================================================
        # 7. Trajectory
        # ==================================================

        pred_traj = self.traj_head(
            query
        )

        pred_traj = pred_traj.reshape(
            B,
            N,
            self.num_modes,
            self.future_steps,
            2,
        )

        # ==================================================
        # 8. Confidence
        # ==================================================

        pred_score = self.score_head(
            query
        )

        pred_score = pred_score.squeeze(-1)

        pred_score = pred_score.reshape(
            B,
            N,
            self.num_modes,
        )

        return {
            "pred_traj": pred_traj,
            "pred_score": pred_score,
        }
