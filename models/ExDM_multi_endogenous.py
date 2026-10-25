"""ExDM7.8 forecasting a named, proper subset of endogenous variables.

Run with --model ExDM_multi_endogenous --features M
--target_cols OT HUFL MUFL --input_dim 4 --output_dim 3 --enc_in 7
--feature_num 7 --fused_exogenous 4 --fused_endogenous 3.

The ETT loader orders channels as [remaining CSV variables, target_cols]
before fitting the training-only scaler. Both x and y use that order.
The inherited pipeline retains decomposition/RevIN, sampled-exogenous
EndL, ExoDiff, WeightMLP, DirectHead and AdaptiveForecastFusion. Training
keeps the original exogenous auxiliary loss; final forecasts and metrics
contain only the endogenous targets, in target_cols order. No future
exogenous observations are supplied to the model during forecasting.
"""

from models.ExDM import EndL, Model as ExDMModel


class Model(ExDMModel):
    def __init__(self, configs):
        targets = list(getattr(configs, 'target_cols', None) or [])
        if len(targets) < 2 or len(set(targets)) != len(targets):
            raise ValueError('Specify at least two distinct endogenous target_cols')
        if configs.features != 'M':
            raise ValueError('Multi-endogenous forecasting requires features=M')
        if configs.output_dim != len(targets) or configs.input_dim < 1:
            raise ValueError('output_dim must match target_cols; at least one exogenous channel is required')
        if (configs.enc_in != configs.input_dim + configs.output_dim
                or configs.feature_num != configs.enc_in
                or configs.fused_exogenous != configs.input_dim
                or configs.fused_endogenous != configs.output_dim):
            raise ValueError('Inconsistent endogenous/exogenous channel dimensions')
        super().__init__(configs)
        self.target_cols = targets
        # The base Namespace omits d_model and endl_* options. Pass the full
        # configuration so this entry uses the EndL settings from its script.
        self.EndL = EndL(configs)

    def forward(self, x, batch_cycle=None, return_components=False):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected [batch, seq_len, enc_in] in exogenous-then-endogenous order')
        return super().forward(x, batch_cycle=batch_cycle, return_components=return_components)
