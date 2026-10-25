import torch
from torch import nn

class DEMA(nn.Module):
    """
    Double Exponential Moving Average (DEMA) block to highlight the trend of time series
    """
    def __init__(self, alpha, beta):
        super(DEMA, self).__init__()
        # self.alpha = nn.Parameter(alpha)    # Learnable alpha
        # self.beta = nn.Parameter(beta)      # Learnable beta
        self.register_buffer('alpha', torch.as_tensor(alpha, dtype=torch.float32))
        self.register_buffer('beta', torch.as_tensor(beta, dtype=torch.float32))

    def forward(self, x):
        # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
        # self.beta.data.clamp_(0, 1)         # Clamp learnable beta to [0, 1]
        alpha = self.alpha.to(device=x.device, dtype=x.dtype)
        beta = self.beta.to(device=x.device, dtype=x.dtype)
        s_prev = x[:, 0, :]
        b = x[:, 1, :] - s_prev
        res = [s_prev.unsqueeze(1)]
        for t in range(1, x.shape[1]):
            xt = x[:, t, :]
            s = alpha * xt + (1 - alpha) * (s_prev + b)
            b = beta * (s - s_prev) + (1 - beta) * b
            s_prev = s
            res.append(s.unsqueeze(1))
        return torch.cat(res, dim=1)
