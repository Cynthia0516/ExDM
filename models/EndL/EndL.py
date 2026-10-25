import torch
import torch.nn as nn


class GatingBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.weight(x)


class EndoExoBackbone(nn.Module):
    """Follow EndL Forcast_multi idea with explicit glob_token, adapted to dual inputs."""

    def __init__(self, seq_len, d_model, endo_channels, exo_channels, t_ff, c_ff, t_dropout, c_dropout, embed_dropout):
        super().__init__()
        self.d_model = d_model
        self.endo_channels = endo_channels
        self.exo_channels = exo_channels

        self.endo_projection = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(embed_dropout),
        )
        self.exo_projection = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(embed_dropout),
        )

        # Keep the original naming style from EndL-main: glob_token.
        self.glob_token = nn.Parameter(torch.ones(1, endo_channels, d_model))

        # TGM: temporal gating on [emb, glob_token].
        self.en_attention = GatingBlock(2 * d_model, t_ff, t_dropout)

        # VGM: variable gating on channel dimension after concatenation to 2 * exo_channels.
        self.ex_attention = GatingBlock(2 * exo_channels, c_ff, c_dropout)

    def forward(self, endo_x, exo_x):
        # endo_x: [B, Ce, L], exo_x: [B, Cm, L]
        bsz, _, _ = endo_x.shape

        endo_emb = self.endo_projection(endo_x)  # [B, Ce, D]
        exo_emb = self.exo_projection(exo_x)  # [B, Cm, D]

        glob_token = self.glob_token.repeat(bsz, 1, 1)  # [B, Ce, D]
        en_emb = torch.cat([endo_emb, glob_token], dim=-1)  # [B, Ce, 2D]
        en_atten = self.en_attention(en_emb)  # [B, Ce, 2D]

        origin_atten = en_atten[:, :, :self.d_model]  # [B, Ce, D]
        glob_atten = en_atten[:, :, self.d_model:]  # [B, Ce, D]

        # Bridge Ce->Cm so exogenous branch can reuse global context from endogenous branch.
        glob_for_exo = glob_atten.mean(dim=1, keepdim=True).repeat(1, self.exo_channels, 1)  # [B, Cm, D]
        ex_emb = torch.cat([exo_emb, glob_for_exo], dim=1)  # [B, 2Cm, D]
        ex_atten = self.ex_attention(ex_emb.permute(0, 2, 1))  # [B, D, 2Cm]

        ex_glob = ex_atten[:, :, self.exo_channels:]  # [B, D, Cm]
        ex_glob = ex_glob.mean(dim=-1, keepdim=True).repeat(1, 1, self.endo_channels)  # [B, D, Ce]

        # Output matches original head expectation: [B, Ce, 2D].
        return torch.cat([origin_atten, ex_glob.permute(0, 2, 1)], dim=-1)


class EndLEndogenous(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        endo_channels,
        exo_channels,
        d_model=128,
        t_ff=128,
        c_ff=128,
        t_dropout=0.0,
        c_dropout=0.0,
        embed_dropout=0.1,
        head_dropout=0.0,
        use_revin=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.endo_channels = endo_channels
        self.exo_channels = exo_channels
        self.use_revin = use_revin

        self.backbone = EndoExoBackbone(
            seq_len=seq_len,
            d_model=d_model,
            endo_channels=endo_channels,
            exo_channels=exo_channels,
            t_ff=t_ff,
            c_ff=c_ff,
            t_dropout=t_dropout,
            c_dropout=c_dropout,
            embed_dropout=embed_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(2 * d_model, pred_len),
        )

    def forward(self, fused_endogenous, selected_exogenous):
        # fused_endogenous: [B, L, endo_channels]
        # selected_exogenous: [B, L, exo_channels]
        if fused_endogenous.ndim != 3 or selected_exogenous.ndim != 3:
            raise ValueError("fused_endogenous and selected_exogenous must be 3D tensors")
        if fused_endogenous.size(1) != self.seq_len or selected_exogenous.size(1) != self.seq_len:
            raise ValueError("sequence length mismatch in EndLEndogenous")
        if fused_endogenous.size(2) != self.endo_channels:
            raise ValueError(f"endogenous channel mismatch: expected {self.endo_channels}, got {fused_endogenous.size(2)}")
        if selected_exogenous.size(2) != self.exo_channels:
            raise ValueError(f"exogenous channel mismatch: expected {self.exo_channels}, got {selected_exogenous.size(2)}")

        if self.use_revin:
            mean = fused_endogenous.mean(dim=1, keepdim=True).detach()
            x_centered = fused_endogenous - mean
            std = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            endo_norm = x_centered / std
            exo_mean = selected_exogenous.mean(dim=1, keepdim=True).detach()
            exo_std = torch.sqrt(torch.var(selected_exogenous - exo_mean, dim=1, keepdim=True, unbiased=False) + 1e-5)
            exo_norm = (selected_exogenous - exo_mean) / exo_std
        else:
            endo_norm = fused_endogenous
            exo_norm = selected_exogenous

        enc = self.backbone(endo_norm.permute(0, 2, 1), exo_norm.permute(0, 2, 1))  # [B, Ce, 2D]
        out = self.head(enc).permute(0, 2, 1)  # [B, H, Ce]

        if self.use_revin:
            out = out * std[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            out = out + mean[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return out
