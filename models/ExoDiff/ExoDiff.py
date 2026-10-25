import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace


class _ExoDiffDenoiser(nn.Module):
    """Channel-wise denoiser used by ExoDiffExogenous.

    Inputs are flattened as [B*C, T], so one set of weights is shared across channels.
    """

    def __init__(self, seq_len: int, pred_len: int, diff_steps: int, hidden_dim: int):
        super().__init__()
        self.pred_len = pred_len
        self.time_embed = nn.Embedding(diff_steps, hidden_dim)
        self.x_proj = nn.Linear(pred_len, hidden_dim)
        self.cond_proj = nn.Linear(seq_len, hidden_dim)
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pred_len),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_feat = self.x_proj(x_t)
        cond_feat = self.cond_proj(cond)
        t_feat = self.time_embed(t)
        h = torch.cat([x_feat, cond_feat, t_feat], dim=-1)
        return self.out(h)


class ExoDiffExogenous(nn.Module):
    """A lightweight ExoDiff-style conditional diffusion model for exogenous forecasting.

    Input:  fused_exogenous [B, L, C]
    Output: fused_exogenous_pred [B, H, C]
    """

    def __init__(self, args: Namespace):
        super().__init__()
        self.seq_len = int(getattr(args, "seq_len", getattr(args, "input_len", 96)))
        self.pred_len = int(getattr(args, "pred_len", 96))
        self.input_dim = int(getattr(args, "input_dim", getattr(args, "feature_num", 1)))
        self.diff_steps = int(getattr(args, "exodiff_steps", getattr(args, "time_steps", 100)))
        hidden_dim = int(getattr(args, "exodiff_hidden_dim", getattr(args, "MLP_hidden_dim", 128)))

        self.register_buffer("betas", torch.linspace(1e-4, 2e-2, self.diff_steps))
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        alphas_cumprod_prev = torch.cat([
            torch.ones(1, device=alphas_cumprod.device),
            alphas_cumprod[:-1],
        ], dim=0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod_prev", torch.sqrt(alphas_cumprod_prev))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod_prev",
            torch.sqrt(1.0 - alphas_cumprod_prev),
        )

        self.denoiser = _ExoDiffDenoiser(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            diff_steps=self.diff_steps,
            hidden_dim=hidden_dim,
        )
        # TimeMachine-style: map fixed history length to prediction length.
        self.target_proj = nn.Linear(self.seq_len, self.pred_len)

    def _normalize(self, x: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return (x - mean) / std, mean, std

    def _extract(self, a: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return a.gather(0, t).view(-1, 1).expand_as(like)

    def _q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self._extract(self.sqrt_alphas_cumprod, t, x_start) * x_start + \
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start) * noise

    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (x_t - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t) * eps) / \
            self._extract(self.sqrt_alphas_cumprod, t, x_t)

    def _resize_last_dim(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        # x: [B, C, L] -> [B, C, target_len]
        if x.shape[-1] == target_len:
            return x
        return F.interpolate(x, size=target_len, mode="linear", align_corners=False)

    def _get_cond_target(self, x_norm: torch.Tensor):
        cond = self._resize_last_dim(x_norm, self.seq_len)
        if cond.shape[-1] >= self.pred_len:
            target = cond[:, :, -self.pred_len:]
        else:
            target = self.target_proj(cond)
        return cond, target

    def forward(self, fused_exogenous: torch.Tensor, task: str = "train") -> torch.Tensor:
        if fused_exogenous.dim() != 3:
            raise ValueError(f"ExoDiffExogenous expects [B, L, C], got {tuple(fused_exogenous.shape)}")

        x_norm, mean, std = self._normalize(fused_exogenous)
        x_norm = x_norm.transpose(1, 2)  # [B, C, L]

        cond, target = self._get_cond_target(x_norm)
        b, c, _ = target.shape

        cond_flat = cond.reshape(b * c, self.seq_len)
        target_flat = target.reshape(b * c, self.pred_len)

        if task == "train":
            t = torch.randint(0, self.diff_steps, (b * c,), device=fused_exogenous.device)
            noise = torch.randn_like(target_flat)
            x_t = self._q_sample(target_flat, t, noise)
            eps = self.denoiser(x_t, t, cond_flat)
            pred = self._predict_x0_from_eps(x_t, t, eps)
        else:
            x_t = torch.randn_like(target_flat)
            for step in reversed(range(self.diff_steps)):
                t = torch.full((b * c,), step, device=fused_exogenous.device, dtype=torch.long)
                eps = self.denoiser(x_t, t, cond_flat)
                x0 = self._predict_x0_from_eps(x_t, t, eps)
                x_t = self._extract(self.sqrt_alphas_cumprod_prev, t, x_t) * x0 + \
                    self._extract(self.sqrt_one_minus_alphas_cumprod_prev, t, x_t) * eps
            pred = x_t

        pred = pred.view(b, c, self.pred_len).transpose(1, 2)  # [B, H, C]

        out_mean = mean[:, :, :self.input_dim].repeat(1, self.pred_len, 1)
        out_std = std[:, :, :self.input_dim].repeat(1, self.pred_len, 1)
        pred = pred * out_std + out_mean
        return pred