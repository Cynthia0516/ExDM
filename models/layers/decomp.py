import torch
from torch import nn

from models.layers.ema import EMA
from models.layers.dema import DEMA


class DECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type, alpha, beta):
        super(DECOMP, self).__init__()
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average




class DECOMP_SeparateChannels(nn.Module):
    """
    实现：为每个通道单独创建分解模块
    """
    def __init__(self, ma_type, alpha, beta, endogenous_dim=-1):
        super(DECOMP_SeparateChannels, self).__init__()
        self.endogenous_dim = endogenous_dim
        self.ma_type = ma_type
        self.alpha = alpha
        self.beta = beta
        
        # 延迟初始化，在第一次forward时根据输入维度创建模块
        self.decomp_modules = None

    def _init_decomp_modules(self, feature_num):
        """根据特征数量初始化DECOMP模块"""
        if self.decomp_modules is not None and len(self.decomp_modules) == feature_num:
            return
            
        self.decomp_modules = nn.ModuleList()
        for _ in range(feature_num):
            self.decomp_modules.append(DECOMP(self.ma_type, self.alpha, self.beta))

    def forward(self, x):
        """
        Args:
            x: input tensor of shape [batch_size, seq_len, feature_num]

        Returns:
            trend_all: [batch_size, seq_len, feature_num]
            seasonal_all: [batch_size, seq_len, feature_num]
        """
        batch_size, seq_len, feature_num = x.shape
        
        # 初始化移动平均模块
        self._init_decomp_modules(feature_num)
        
        # 对每个特征通道进行分解
        seasonal_components = []
        trend_components = []
        
        for i in range(feature_num):
            channel_data = x[:, :, i:i+1]  # [batch_size, seq_len, 1]
            decomp_module= self. decomp_modules[i]
            seasonal, trend = decomp_module(channel_data)
            
            seasonal_components.append(seasonal)
            trend_components.append(trend)
        
        # 合并所有通道
        seasonal_all = torch.cat(seasonal_components, dim=-1)  # [batch_size, seq_len, feature_num]
        trend_all = torch.cat(trend_components, dim=-1)        # [batch_size, seq_len, feature_num]
        
        return trend_all, seasonal_all