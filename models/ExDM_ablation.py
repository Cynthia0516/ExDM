import torch
import torch.nn as nn
from argparse import Namespace

from models.ExDM import (
    Datadecomp,
    DirectForecastHead,
    EndL,
    ExoDiff,
    WeightMLP,
)


class TwoBranchAdaptiveForecastFusion(nn.Module):
    def __init__(self, pred_len, output_dim, hidden_dim=64, dropout=0.05, init_direct_weight=0.8):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.num_candidates = 2

        init_direct_weight = min(max(float(init_direct_weight), 0.05), 0.95)
        probs = torch.tensor([init_direct_weight, 1.0 - init_direct_weight], dtype=torch.float32)
        probs = probs / probs.sum()
        self.base_logits = nn.Parameter(torch.log(probs).view(1, 1, 1, self.num_candidates))

        feature_dim = self.num_candidates + 3
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_candidates),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        horizon = torch.linspace(0.0, 1.0, pred_len, dtype=torch.float32).view(1, pred_len, 1, 1)
        self.register_buffer('horizon_pos', horizon, persistent=False)

    def forward(self, raw_endogenous, direct_output, endl_output):
        candidates = torch.stack([direct_output, endl_output], dim=-1)
        last = raw_endogenous[:, -1:, :].unsqueeze(-1)
        scale = raw_endogenous.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5).unsqueeze(-1)
        cand_norm = (candidates - last) / scale

        trend_window = min(24, max(2, raw_endogenous.size(1) // 4))
        trend_base = raw_endogenous[:, -trend_window:-trend_window + 1, :]
        slope = (raw_endogenous[:, -1:, :] - trend_base) / float(trend_window - 1)
        slope_norm = (slope.unsqueeze(-1) / scale).expand(-1, self.pred_len, -1, 1)
        level_norm = ((raw_endogenous[:, -1:, :] - raw_endogenous.mean(dim=1, keepdim=True)).unsqueeze(-1) / scale)
        level_norm = level_norm.expand(-1, self.pred_len, -1, 1)
        horizon_pos = self.horizon_pos.to(raw_endogenous.device).expand(raw_endogenous.size(0), -1, self.output_dim, -1)

        gate_features = torch.cat([cand_norm, slope_norm, level_norm, horizon_pos], dim=-1)
        logits = self.base_logits + self.gate(gate_features)
        weights = torch.softmax(logits, dim=-1)
        fused = (candidates * weights).sum(dim=-1)
        return fused, weights


class Model(nn.Module):
    """Ablation variants for the ExDM forecasting head."""

    VALID_MODES = {'endl_only', 'direct_endl', 'direct_exodiff'}

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.fused_exogenous = getattr(configs, 'input_dim', getattr(configs, 'fused_exogenous', 1))
        self.fused_endogenous = getattr(configs, 'output_dim', getattr(configs, 'fused_endogenous', 1))
        self.ablation_mode = getattr(configs, 'ablation_mode', 'endl_only')
        if self.ablation_mode not in self.VALID_MODES:
            raise ValueError(f"ablation_mode must be one of {sorted(self.VALID_MODES)}, got {self.ablation_mode}")

        if self.fused_exogenous + self.fused_endogenous != self.enc_in:
            raise ValueError(
                f"input_dim + output_dim must equal enc_in ({self.enc_in}), got {self.fused_exogenous + self.fused_endogenous}"
            )

        endl_args = Namespace(
            input_len=configs.seq_len,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            feature_num=configs.feature_num,
            input_dim=self.fused_exogenous,
            feature_dim=self.fused_endogenous,
            fused_endogenous_dim=self.fused_endogenous,
            sample_alpha=getattr(configs, 'sample_alpha', 0.3),
            sample_p=getattr(configs, 'sample_p', 0.1),
            time_steps=getattr(configs, 'time_steps', 1000),
            MLP_hidden_dim=getattr(configs, 'MLP_hidden_dim', 128),
            device=getattr(configs, 'device', 'cuda'),
            d_model=getattr(configs, 'd_model', 128),
            endl_t_ff=getattr(configs, 'endl_t_ff', getattr(configs, 'd_model', 128)),
            endl_c_ff=getattr(configs, 'endl_c_ff', getattr(configs, 'd_model', 128)),
            endl_t_dropout=getattr(configs, 'endl_t_dropout', 0.0),
            endl_c_dropout=getattr(configs, 'endl_c_dropout', 0.0),
            endl_embed_dropout=getattr(configs, 'endl_embed_dropout', 0.1),
            endl_head_dropout=getattr(configs, 'endl_head_dropout', 0.0),
        )

        self.datadecomp = Datadecomp(
            configs=configs,
            sizes=[configs.seq_len, getattr(configs, 'label_len', 0), configs.pred_len],
            ma_type=getattr(configs, 'ma_type', 'ema'),
            feature_num=configs.enc_in,
            in_channels_seasonal=self.fused_exogenous,
            in_channels_trend=self.fused_exogenous,
            out_channels=self.fused_endogenous,
            endogenous_dim=-1,
        )
        self.EndL = EndL(endl_args)

        self.DirectHead = DirectForecastHead(
            self.seq_len,
            self.pred_len,
            self.fused_endogenous,
            dropout=getattr(configs, 'direct_dropout', 0.05),
            individual=True,
        )

        self.ExoDiff = ExoDiff(configs, input_dim=self.fused_exogenous)
        self.WeightMLP = WeightMLP(
            self.fused_exogenous,
            self.fused_endogenous,
            self.seq_len,
            self.pred_len,
            hidden_dim=getattr(configs, 'dff', 64),
        )

        fusion_hidden_dim = getattr(configs, 'fusion_hidden_dim', max(16, getattr(configs, 'd_model', 128) // 2))
        fusion_dropout = getattr(configs, 'fusion_dropout', getattr(configs, 'direct_dropout', 0.05))
        direct_weight = float(getattr(configs, 'direct_weight', 0.8))
        self.DirectEndLFusion = TwoBranchAdaptiveForecastFusion(
            self.pred_len,
            self.fused_endogenous,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
            init_direct_weight=direct_weight,
        )
    def _denorm_endogenous_outputs(self, outputs):
        if not self.datadecomp.revin:
            return outputs
        endo_slice = slice(self.fused_exogenous, self.fused_exogenous + self.fused_endogenous)
        return self.datadecomp.revin_denorm(outputs, endo_slice)

    def forward(self, x, batch_cycle=None, return_components=False):
        raw_endogenous = x[:, :, self.fused_exogenous:self.fused_exogenous + self.fused_endogenous]
        direct_output = self.DirectHead(raw_endogenous)
        fused_exogenous, fused_endogenous, trend_endogenous, seasonal_endogenous = self.datadecomp(x)

        fused_endogenous_pred = None
        fused_endogenous_pred_list = None
        fused_endogenous1 = None
        sampled_exogenous = None
        sample_m = None
        sample_n = None
        fused_exogenous_pred = None
        weightmlp_output = None
        fusion_weights = None

        if self.ablation_mode in ('endl_only', 'direct_endl'):
            fused_endogenous_pred, fused_endogenous1, sampled_exogenous, fused_endogenous_pred_list, sample_m, sample_n = self.EndL(
                fused_endogenous,
                fused_exogenous,
                batch_cycle,
                return_fused=True,
            )
            fused_endogenous_pred = self._denorm_endogenous_outputs(fused_endogenous_pred)
            if fused_endogenous_pred_list is not None:
                fused_endogenous_pred_list = self._denorm_endogenous_outputs(fused_endogenous_pred_list)

        if self.ablation_mode == 'endl_only':
            output = fused_endogenous_pred
        elif self.ablation_mode == 'direct_endl':
            output, fusion_weights = self.DirectEndLFusion(raw_endogenous, direct_output, fused_endogenous_pred)
        else:
            fused_exogenous_pred = self.ExoDiff(fused_exogenous)
            weightmlp_output = self.WeightMLP(fused_exogenous, fused_exogenous_pred, direct_output)
            output = weightmlp_output

        if return_components:
            return output, {
                'ablation_mode': self.ablation_mode,
                'fused_exogenous': fused_exogenous,
                'fused_exogenous_pred': fused_exogenous_pred,
                'fused_endogenous': fused_endogenous,
                'fused_endogenous1': fused_endogenous1,
                'sampled_exogenous': sampled_exogenous,
                'fused_endogenous_pred': fused_endogenous_pred,
                'fused_endogenous_pred_list': fused_endogenous_pred_list,
                'weightmlp_output': weightmlp_output,
                'direct_output': direct_output,
                'fusion_weights': fusion_weights,
                'sample_m': sample_m,
                'sample_n': sample_n,
                'trend_endogenous': trend_endogenous,
                'seasonal_endogenous': seasonal_endogenous,
            }

        return output
