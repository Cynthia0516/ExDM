import torch
import torch.nn as nn
import argparse
import math
import os
import time
import numpy as np
from models.layers.decomp import DECOMP
from models.layers.decomp import DECOMP_SeparateChannels
from models.layers.revin import RevIN
from models.layers.firstfusion import TrendSeasonalFusion
from models.layers.firstfusion import MultiVariateFusionNetwork
from models.layers.Conv_Blocks import Inception_Block_V2
from models.EndL.EndL import EndLEndogenous
from models.ExoDiff.ExoDiff import ExoDiffExogenous
from argparse import Namespace

class Datadecomp(nn.Module):
    def __init__(self, configs,sizes,ma_type,feature_num,in_channels_seasonal,in_channels_trend,out_channels,endogenous_dim=-1):
        super(Datadecomp, self).__init__()

        # Parameters参数
        c_in = configs.enc_in       # input channels
        self.input_dim = getattr(configs, 'input_dim', in_channels_trend)
        self.output_dim = getattr(configs, 'output_dim', out_channels)

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)

        # Moving Average Decomposition
        self.ma_type = getattr(configs, 'ma_type', ma_type)
        alpha = getattr(configs, 'alpha', 0.3)       # smoothing factor for EMA (Exponential Moving Average) EMA（指数移动平均线）的平滑系数
        beta = getattr(configs, 'beta', 0.3)         # smoothing factor for DEMA (Double Exponential Moving Average)双指数移动平均线（DEMA）的平滑系数

        self.decomp = DECOMP_SeparateChannels(self.ma_type, alpha, beta,)

        self.endogenous_dim = endogenous_dim
        self.alpha = alpha
        self.beta = beta

        self.exo_fusion = TrendSeasonalFusion(
            in_channels_trend=self.input_dim,
            in_channels_seasonal=self.input_dim,
            out_channels=self.input_dim
        )
        self.endo_fusion = TrendSeasonalFusion(
            in_channels_trend=self.output_dim,
            in_channels_seasonal=self.output_dim,
            out_channels=self.output_dim
        )

    def forward(self, x):
        # x: [Batch, Input, Channel]

        if x.shape[-1] != self.input_dim + self.output_dim:
            raise ValueError(
                f"channels must equal input_dim+output_dim ({self.input_dim + self.output_dim}), got {x.shape[-1]}"
            )

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':   # If no decomposition, directly pass the input to the network
            trend_all = x
            seasonal_all = x
        else:
            trend_all, seasonal_all = self.decomp(x)

        trend_exogenous = trend_all[:, :, :self.input_dim]
        seasonal_exogenous = seasonal_all[:, :, :self.input_dim]
        trend_endogenous = trend_all[:, :, self.input_dim:self.input_dim + self.output_dim]
        seasonal_endogenous = seasonal_all[:, :, self.input_dim:self.input_dim + self.output_dim]

        fused_exogenous = self.exo_fusion(trend_exogenous, seasonal_exogenous)
        fused_endogenous = self.endo_fusion(trend_endogenous, seasonal_endogenous)

        return fused_exogenous, fused_endogenous, trend_endogenous, seasonal_endogenous

    def revin_denorm(self, x, channel_slice):
        if not self.revin:
            return x
        rev = self.revin_layer
        if not hasattr(rev, 'stdev'):
            return x

        if x.dim() == 3:
            if rev.affine:
                weight = rev.affine_weight[channel_slice].view(1, 1, -1)
                bias = rev.affine_bias[channel_slice].view(1, 1, -1)
                x = (x - bias) / (weight + rev.eps * rev.eps)
            stdev = rev.stdev[:, :, channel_slice]
            base = rev.last[:, :, channel_slice] if rev.subtract_last else rev.mean[:, :, channel_slice]
            return x * stdev + base

        if x.dim() == 4:
            if rev.affine:
                weight = rev.affine_weight[channel_slice].view(1, 1, 1, -1)
                bias = rev.affine_bias[channel_slice].view(1, 1, 1, -1)
                x = (x - bias) / (weight + rev.eps * rev.eps)
            stdev = rev.stdev[:, :, channel_slice].unsqueeze(0)
            base = rev.last[:, :, channel_slice] if rev.subtract_last else rev.mean[:, :, channel_slice]
            return x * stdev + base.unsqueeze(0)

        return x
        


class EndL(nn.Module):
    def __init__(self, endo_args):
        super(EndL, self).__init__()
        self.input_dim = int(getattr(endo_args, 'input_dim', 1))
        self.sample_alpha = float(getattr(endo_args, 'sample_alpha', getattr(endo_args, 'alpha', 0.3)))
        self.sample_p = float(getattr(endo_args, 'sample_p', getattr(endo_args, 'p', 0.1)))
        self.output_dim = int(getattr(endo_args, 'output_dim', getattr(endo_args, 'fused_endogenous_dim', 1)))
        self.sample_m = max(1, min(int(math.ceil(self.sample_alpha * self.input_dim)), self.input_dim))
        # 内生变量预测：fused_endogenous -> TGM, selected_exogenous -> VGM
        self.endl = EndLEndogenous(
            seq_len=endo_args.seq_len,
            pred_len=endo_args.pred_len,
            endo_channels=self.output_dim,
            exo_channels=self.sample_m,
            d_model=getattr(endo_args, 'd_model', 128),
            t_ff=getattr(endo_args, 'endl_t_ff', getattr(endo_args, 'd_model', 128)),
            c_ff=getattr(endo_args, 'endl_c_ff', getattr(endo_args, 'd_model', 128)),
            t_dropout=getattr(endo_args, 'endl_t_dropout', 0.0),
            c_dropout=getattr(endo_args, 'endl_c_dropout', 0.0),
            embed_dropout=getattr(endo_args, 'endl_embed_dropout', 0.1),
            head_dropout=getattr(endo_args, 'endl_head_dropout', 0.0),
            use_revin=bool(getattr(endo_args, 'use_revin', True)),
        )

    def _sample_exogenous_list(self, fused_exogenous):
        # m = ceil(alpha * input_dim)
        input_dim = fused_exogenous.size(-1)
        m = self.sample_m
        if m > input_dim:
            m = input_dim

        # n >= ln(p) / ln(1 - m / input_dim), then ceil
        # Clamp p into (0, 1) for numerical stability.
        p = min(max(self.sample_p, 1e-8), 1 - 1e-8)
        ratio = 1.0 - float(m) / float(input_dim)
        if ratio <= 0:
            n = 1
        else:
            n = int(math.ceil(math.log(p) / math.log(ratio)))
            n = max(1, n)

        selected_exogenous_list = []
        base_idx = torch.arange(input_dim, device=fused_exogenous.device)
        for i in range(n):
            # 无放回抽样 m 个变量: [m]
            if self.training:
                idx = torch.randperm(input_dim, device=fused_exogenous.device)[:m]
            else:
                idx = torch.roll(base_idx, shifts=-i)[:m]
            selected_exogenous_list.append(fused_exogenous[:, :, idx])  # [B, L, m]

        return selected_exogenous_list, m, n

    def forward(self, fused_endogenous, fused_exogenous, cycle_index=None, return_fused=False):
        # fused_endogenous: [B, L, output_dim]
        # fused_exogenous: [B, L, input_dim]
        selected_exogenous_list, m, n = self._sample_exogenous_list(fused_exogenous)

        # n 次抽样对应 n 次 EndL；每次输入不同的 selected_exogenous，但共享 fused_endogenous。
        fused_endogenous_pred_list = [
            self.endl(fused_endogenous, selected_exogenous_list[i])
            for i in range(n)
        ]
        fused_endogenous_pred_stack = torch.stack(fused_endogenous_pred_list, dim=0)  # [n, B, H, output_dim]
        fused_endogenous_pred = fused_endogenous_pred_stack.mean(dim=0)  # [B, H, output_dim]

        if return_fused:
            selected_exogenous_stack = torch.stack(selected_exogenous_list, dim=0)  # [n, B, L, m]
            return fused_endogenous_pred, fused_endogenous, selected_exogenous_stack, fused_endogenous_pred_stack, m, n
        return fused_endogenous_pred


    #Exogenous Variable Image Diffusion Model

class ExoDiff(nn.Module):
    def __init__(self, args, input_dim, feature_num=None):
        super(ExoDiff, self).__init__()
        # 外生变量扩散：fused_exogenous -> ExoDiff -> fused_exogenous_pred
        self.input_dim = getattr(args, 'input_dim', input_dim)
        self.LayerNorm = nn.LayerNorm(self.input_dim)

        exodiff_args = Namespace(
            input_len=getattr(args, 'seq_len', getattr(args, 'input_len', None)),
            seq_len=getattr(args, 'seq_len', getattr(args, 'input_len', None)),
            pred_len=getattr(args, 'pred_len', None),
            feature_num=getattr(args, 'feature_num', self.input_dim),
            time_steps=getattr(args, 'time_steps', 1000),
            MLP_hidden_dim=getattr(args, 'MLP_hidden_dim', 128),
            input_dim=self.input_dim,
            exodiff_steps=getattr(args, 'exodiff_steps', getattr(args, 'time_steps', 100)),
            exodiff_hidden_dim=getattr(args, 'exodiff_hidden_dim', getattr(args, 'MLP_hidden_dim', 128)),
        )
        self.exodiff = ExoDiffExogenous(exodiff_args)

    def forward(self, fused_exogenous):
        # fused_exogenous: [B, L, input_dim]
        task = 'train' if self.training else 'eval'
        fused_exogenous_pred = self.exodiff(fused_exogenous, task=task)
        fused_exogenous_pred = self.LayerNorm(fused_exogenous_pred)
        # fused_exogenous_pred: [B, H, input_dim]
        return fused_exogenous_pred


class WeightMLP(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, pred_len, hidden_dim=64):
        super(WeightMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 在时间维做 MLP: [B, C, L+H] -> [B, C, H]
        self.temporal_mlp = nn.Sequential(
            nn.Linear(self.seq_len + self.pred_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.pred_len)
        )

        # 对 W1 在通道维做修正: [B, H, input_dim] -> [B, H, output_dim]
        self.channel_correct = nn.Linear(self.input_dim, self.output_dim)
        self.residual_correct = nn.Linear(self.input_dim, self.output_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, fused_exogenous, fused_exogenous_pred, fused_endogenous_pred):
        # 1) 按时间维拼接，语义 [B, L+H, input_dim]
        exo_all = torch.cat([fused_exogenous, fused_exogenous_pred], dim=1)

        # 2) 对第二维(时间维)做 MLP，得到 W1: [B, H, input_dim]
        w1 = self.temporal_mlp(exo_all.transpose(1, 2)).transpose(1, 2)

        # 3) 通道修正，得到 W2: [B, H, output_dim]
        gate_delta = torch.tanh(self.channel_correct(w1))
        residual = torch.tanh(self.residual_correct(w1))

        # 4) 对 EndL 预测做残差校准。原来的 sigmoid 乘法只能衰减预测，
        #    这里允许轻量放大、压缩和加性修正，提升长预测区间的表达能力。
        scale = torch.clamp(self.residual_scale, 0.0, 1.0)
        return fused_endogenous_pred + scale * (gate_delta * fused_endogenous_pred + residual)


class DirectForecastHead(nn.Module):
    def __init__(self, seq_len, pred_len, output_dim, dropout=0.05, individual=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.individual = individual
        self.dropout = nn.Dropout(dropout)
        if individual:
            self.linears = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(output_dim)])
        else:
            self.linear = nn.Linear(seq_len, pred_len)
        self.trend_scale = nn.Parameter(torch.zeros(1, 1, output_dim))
        horizon = torch.arange(1, pred_len + 1, dtype=torch.float32).view(1, pred_len, 1)
        self.register_buffer('horizon_index', horizon, persistent=False)

    def forward(self, x_endo):
        # NLinear-style forecast on the target/endogenous channels in dataset scale.
        seq_last = x_endo[:, -1:, :].detach()
        x = self.dropout(x_endo - seq_last).transpose(1, 2)
        if self.individual:
            outs = [self.linears[i](x[:, i, :]).unsqueeze(1) for i in range(self.output_dim)]
            out = torch.cat(outs, dim=1).transpose(1, 2)
        else:
            out = self.linear(x).transpose(1, 2)
        trend_window = min(24, max(2, self.seq_len // 4))
        trend_base = x_endo[:, -trend_window:-trend_window + 1, :]
        slope = (x_endo[:, -1:, :] - trend_base) / float(trend_window - 1)
        trend = self.horizon_index.to(x_endo.device) * slope * self.trend_scale
        return out + seq_last + trend


class AdaptiveForecastFusion(nn.Module):
    def __init__(self, pred_len, output_dim, hidden_dim=64, dropout=0.05, init_direct_weight=0.8):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.num_candidates = 3

        init_direct_weight = min(max(float(init_direct_weight), 0.05), 0.95)
        endl_weight = 0.02
        weightmlp_weight = max(1.0 - init_direct_weight - endl_weight, 1e-3)
        probs = torch.tensor([init_direct_weight, weightmlp_weight, endl_weight], dtype=torch.float32)
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

    def forward(self, raw_endogenous, direct_output, weightmlp_output, endl_output):
        candidates = torch.stack([direct_output, weightmlp_output, endl_output], dim=-1)
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





###以下是AI
class Model(nn.Module):
    """Training entrypoint used by exp_main.py: ExDM.Model(args)."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.fused_exogenous = getattr(configs, 'input_dim', getattr(configs, 'fused_exogenous', 1))
        self.fused_endogenous = getattr(configs, 'output_dim', getattr(configs, 'fused_endogenous', 1))
        cdpm_args = Namespace(
            input_len=configs.seq_len,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            feature_num=configs.feature_num,
            input_dim=self.fused_exogenous,
            feature_dim=self.fused_endogenous,
            fused_endogenous_dim=self.fused_endogenous,
            sample_alpha=getattr(configs, 'sample_alpha', 0.3),
            sample_p=getattr(configs, 'sample_p', 0.1),
            fusion_kernel_size=getattr(configs, "fusion_kernel_size", 3),
            time_steps=getattr(configs, "time_steps", 1000),
            MLP_hidden_dim=getattr(configs, "MLP_hidden_dim", 128),
            emb_dim=getattr(configs, "emb_dim", 128),
            scheduler=getattr(configs, "scheduler", "linear"),
            patch_size=getattr(configs, "patch_size", 8),
            device=getattr(configs, "device", "cuda"),
        )
        # Opt in for sensitivity experiments while preserving existing scripts.
        # Forward only t_ff; other EndL settings keep their previous behavior.
        if getattr(configs, 'enable_endl_t_ff', 0):
            cdpm_args.endl_t_ff = getattr(configs, 'endl_t_ff', None) or 256
        if self.fused_exogenous + self.fused_endogenous != self.enc_in:
            raise ValueError(
                f"input_dim + output_dim must equal enc_in ({self.enc_in}), got {self.fused_exogenous + self.fused_endogenous}"
            )

        #hidden_dim = getattr(configs, 'n1', 256)
        #dropout = getattr(configs, 'dropout', 0.0)

        # Channel-wise temporal projection: [B, C, seq_len] -> [B, C, pred_len]
        # self.backbone = nn.Sequential(
        #     nn.Linear(configs.seq_len, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, configs.pred_len),
        # )
        self.EndL = EndL(cdpm_args)
        self.ExoDiff = ExoDiff(configs, input_dim=self.fused_exogenous)

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

        # WeightMLP: 输入 (fused_exogenous, fused_exogenous_pred)，输出用于修正 fused_endogenous_pred
        self.WeightMLP = WeightMLP(
            self.fused_exogenous,
            self.fused_endogenous,
            self.seq_len,
            self.pred_len,
            hidden_dim=getattr(configs, 'dff', 64)
        )
        self.DirectHead = DirectForecastHead(
            self.seq_len,
            self.pred_len,
            self.fused_endogenous,
            dropout=getattr(configs, 'direct_dropout', 0.05),
            individual=True,
        )
        self.direct_weight = float(getattr(configs, 'direct_weight', 0.8))
        self.ForecastFusion = AdaptiveForecastFusion(
            self.pred_len,
            self.fused_endogenous,
            hidden_dim=getattr(configs, 'fusion_hidden_dim', max(16, getattr(configs, 'd_model', 128) // 2)),
            dropout=getattr(configs, 'fusion_dropout', getattr(configs, 'direct_dropout', 0.05)),
            init_direct_weight=self.direct_weight,
        )

    def forward(self, x, batch_cycle=None, return_components=False):
        # x: [B, L, C]
        raw_endogenous = x[:, :, self.fused_exogenous:self.fused_exogenous + self.fused_endogenous]
        direct_output = self.DirectHead(raw_endogenous)
        fused_exogenous, fused_endogenous, trend_endogenous, seasonal_endogenous = self.datadecomp(x)

        fused_endogenous_pred, fused_endogenous1, sampled_exogenous, fused_endogenous_pred_list, sample_m, sample_n = self.EndL(
            fused_endogenous,
            fused_exogenous,
            batch_cycle,
            return_fused=True
        )  # [B, H, output_dim]
        fused_exogenous_pred = self.ExoDiff(fused_exogenous)      # [B, H, input_dim]

        # 根据外生历史+外生预测得到 W1/W2，并修正内生预测
        x = self.WeightMLP(
            fused_exogenous,
            fused_exogenous_pred,
            fused_endogenous_pred
        )
        weightmlp_output = x
        if self.datadecomp.revin:
            endo_slice = slice(self.fused_exogenous, self.fused_exogenous + self.fused_endogenous)
            fused_endogenous_pred = self.datadecomp.revin_denorm(fused_endogenous_pred, endo_slice)
            if fused_endogenous_pred_list is not None:
                fused_endogenous_pred_list = self.datadecomp.revin_denorm(fused_endogenous_pred_list, endo_slice)
            weightmlp_output = self.datadecomp.revin_denorm(weightmlp_output, endo_slice)

        x, fusion_weights = self.ForecastFusion(
            raw_endogenous,
            direct_output,
            weightmlp_output,
            fused_endogenous_pred
        )
        if return_components:
            return x, {
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

        return x
