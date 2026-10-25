import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.layers.decomp import DECOMP_SeparateChannels
from models.layers.Conv_Blocks import Inception_Block_V2
class TrendSeasonalFusion(nn.Module):
    """
    使用1D卷积融合趋势项和季节项
    """
    def __init__(self, 
                 in_channels_trend: int,
                 in_channels_seasonal: int,
                 out_channels: int,
                 kernel_size: int = 5,
                 fusion_method: str = 'concat_conv'):
        super(TrendSeasonalFusion, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd to keep sequence length unchanged, got {kernel_size}"
            )
        self.fusion_method = fusion_method
        self.out_channels = out_channels
        
        if fusion_method == 'concat_conv':
            # 拼接后卷积融合
            total_in_channels = in_channels_trend + in_channels_seasonal
            self.fusion_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=total_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    padding_mode='zeros'
                ),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels)
            )
            
        elif fusion_method == 'separate_conv_add':
            # 分别卷积后相加
            self.trend_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels_trend,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2
                ),
                nn.ReLU()
            )
            self.seasonal_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels_seasonal,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2
                ),
                nn.ReLU()
            )
            
        elif fusion_method == 'weighted_sum':
            # 加权求和融合
            self.trend_weight = nn.Parameter(torch.tensor(0.7))
            self.seasonal_weight = nn.Parameter(torch.tensor(0.3))
            # 确保维度匹配的投影层
            if in_channels_trend != out_channels:
                self.trend_proj = nn.Conv1d(in_channels_trend, out_channels, 1)
            else:
                self.trend_proj = nn.Identity()
                
            if in_channels_seasonal != out_channels:
                self.seasonal_proj = nn.Conv1d(in_channels_seasonal, out_channels, 1)
            else:
                self.seasonal_proj = nn.Identity()
                
            self.output_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2
                ),
                nn.ReLU()
            )

    def forward(self, trend: torch.Tensor, seasonal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trend: [batch_size, seq_len, channels_trend]
            seasonal: [batch_size, seq_len, channels_seasonal]
        
        Returns:
            fused: [batch_size, seq_len, out_channels]
        """
        batch_size, seq_len = trend.shape[0], trend.shape[1]
        
        # 确保输入是 [batch_size, channels, seq_len] 格式
        if trend.dim() == 3:
            trend_t = trend.transpose(1, 2)  # [batch, channels, seq_len]
        else:
            trend_t = trend
            
        if seasonal.dim() == 3:
            seasonal_t = seasonal.transpose(1, 2)
        else:
            seasonal_t = seasonal
            
        if self.fusion_method == 'concat_conv':
            # 拼接趋势和季节项
            combined = torch.cat([trend_t, seasonal_t], dim=1)
            fused = self.fusion_conv(combined)
            
        elif self.fusion_method == 'separate_conv_add':
            # 分别卷积后相加
            trend_processed = self.trend_conv(trend_t)
            seasonal_processed = self.seasonal_conv(seasonal_t)
            fused = trend_processed + seasonal_processed
            
        elif self.fusion_method == 'weighted_sum':
            # 加权求和
            trend_proj = self.trend_proj(trend_t)
            seasonal_proj = self.seasonal_proj(seasonal_t)
            
            # 应用可学习权重
            weighted_trend = self.trend_weight * trend_proj
            weighted_seasonal = self.seasonal_weight * seasonal_proj
            combined = weighted_trend + weighted_seasonal
            fused = self.output_conv(combined)
        
        # 转置回 [batch_size, seq_len, channels] 格式
        fused = fused.transpose(1, 2)
            
        return fused


class MultiVariateFusionNetwork(nn.Module):
    """
    多变量时序融合网络：分别融合内生变量和外生变量
    """
    def __init__(self, 
                 feature_num: int,  # 新增：总特征维度
                 endogenous_trend_dim: int = 1,
                 endogenous_seasonal_dim: int = 1,
                 #exogenous_trend_dim: int = int - 1,##应该是n-1
                 #exogenous_seasonal_dim: int = int - 1,##应该是n-1
                 fused_endogenous_dim: int = 1,
                 fused_exogenous_dim: Optional[int] = None,
                 kernel_size: int = 5,
                 fusion_method: str = 'concat_conv',
                 use_final_fusion: bool = True):
        super(MultiVariateFusionNetwork, self).__init__()
        
        self.use_final_fusion = use_final_fusion
        self.fused_endogenous_dim = fused_endogenous_dim
        
        # 计算外生变量维度（总特征数-1）
        exogenous_trend_dim = feature_num - 1
        exogenous_seasonal_dim = feature_num - 1

        # 如果未指定 fused_exogenous_dim，默认也设为 num_features - 1
        if fused_exogenous_dim is None:
            fused_exogenous_dim = feature_num - 1
        self.fused_exogenous_dim = fused_exogenous_dim
        # 内生变量融合模块
        self.endogenous_fusion = TrendSeasonalFusion(
            in_channels_trend=endogenous_trend_dim,
            in_channels_seasonal=endogenous_seasonal_dim,
            out_channels=fused_endogenous_dim,
            kernel_size=kernel_size,
            fusion_method=fusion_method
        )
        
        # 外生变量融合模块
        self.exogenous_fusion = TrendSeasonalFusion(
            in_channels_trend=exogenous_trend_dim,
            in_channels_seasonal=exogenous_seasonal_dim,
            out_channels=fused_exogenous_dim,
            kernel_size=kernel_size,
            fusion_method=fusion_method
        )
        
        
        

    def forward(self, decomposition_output: dict) -> dict:
        """
        Args:
            decomposition_output: DECOMP_SeparateChannels的输出字典
        
        Returns:
            dict: 包含融合后的内生变量和外生变量
        """
        # 提取趋势和季节分量
        trend_endo = decomposition_output['trend_endogenous']  # [batch, seq_len, 1]
        seasonal_endo = decomposition_output['seasonal_endogenous']  # [batch, seq_len, 1]
        trend_exo = decomposition_output['trend_exogenous']  # [batch, seq_len, n_exo]
        seasonal_exo = decomposition_output['seasonal_exogenous']  # [batch, seq_len, n_exo]
        
        # 融合内生变量
        fused_endogenous = self.endogenous_fusion(trend_endo, seasonal_endo)
        # [batch, seq_len, fused_endogenous_dim]
        
        # 融合外生变量
        fused_exogenous = self.exogenous_fusion(trend_exo, seasonal_exo)
        # [batch, seq_len, fused_exogenous_dim]
        
        return fused_endogenous,fused_exogenous



