import argparse
import csv
import os
import torch
from exp.exp_main import Exp_Main
import random

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    # RANDOM SEED
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
    
    # BASIC CONFIG
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='ExDM',
                        help='model name, options: [ExDM, ExDM_ablation, ExDM_multi_endogenous]')
    parser.add_argument('--ablation_mode', type=str, default='full',
                        choices=['full', 'endl_only', 'direct_endl', 'direct_exodiff'],
                        help='ablation variant used when --model ExDM_ablation')
    parser.add_argument('--model_id_name', type=str, required=False, default='custom', help='model id name')

    # DATALOADER
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--target_cols', nargs='+', default=None,
                        help='Ordered endogenous columns for ExDM_multi_endogenous (default: OT HUFL MUFL)')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # FORECASTING TASK
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    #parser.add_argument('--n1',type=int,default=256,help='First Embedded representation')
    parser.add_argument('--n2',type=int,default=128,help='Second Embedded representation')


    # METHOD
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--ch_ind', type=int, default=1, help='Channel Independence; True 1 False 0')
    parser.add_argument('--residual', type=int, default=1, help='Residual Connection; True 1 False 0')
    parser.add_argument('--d_model', type=int, default=256, help='model dimension')
    parser.add_argument('--d_state', type=int, default=256, help='d_state parameter of Mamba')
    parser.add_argument('--dconv', type=int, default=2, help='d_conv parameter of Mamba')
    parser.add_argument('--e_fact', type=int, default=1, help='expand factor parameter of Mamba')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') #Use this hyperparameter as the number of channels
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--feature_num', type=int, required=True, help='feature num of the dataset')#缺少报错，不是默认
    parser.add_argument('--input_dim', type=int, required=True, help=' input dimension of WeightMLP')#fused_exogenous
    parser.add_argument('--output_dim', type=int, required=True, help='output dimension of the dataset')#fused_endogenous
    parser.add_argument('--fused_exogenous', type=int, required=True, help='fused_exogenous dimension')
    parser.add_argument('--fused_endogenous', type=int, required=True, help='fused_endogenous dimension')
    parser.add_argument('--interpdn_patch_len', type=int, default=16,
                        help='patch length for interPDN endogenous branch')
    parser.add_argument('--interpdn_stride', type=int, default=8,
                        help='patch stride for interPDN endogenous branch')
    parser.add_argument('--interpdn_padding_patch', type=str, default='end', choices=['end', 'none'],
                        help='patch padding mode for interPDN endogenous branch')
    parser.add_argument('--interpdn_support_size', type=int, default=25,
                        help='number of support points for interPDN distribution head')
    parser.add_argument('--ma_type', type=str, default='ema', choices=['ema', 'dema', 'reg'],
                        help='moving average decomposition type')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='smoothing factor for EMA/DEMA')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='second smoothing factor for DEMA')
    # EndL sampling hyper-parameters
    parser.add_argument('--sample_alpha', type=float, default=0.3,
                        help='sampling ratio alpha for m=ceil(alpha*input_dim)')
    parser.add_argument('--sample_p', type=float, default=0.1,
                        help='target probability p for n>=ln(p)/ln(1-m/input_dim)')
    # EndL hyper-parameters for EndL branch
    parser.add_argument('--endl_t_ff', type=int, default=None,
                        help='temporal FFN hidden dim in EndL (default: d_model)')
    parser.add_argument('--enable_endl_t_ff', type=int, default=0, choices=[0, 1],
                        help='forward endl_t_ff to the full ExDM model; enabled by sensitivity runs')
    parser.add_argument('--endl_c_ff', type=int, default=None,
                        help='channel FFN hidden dim in EndL (default: d_model)')
    parser.add_argument('--endl_t_dropout', type=float, default=0.0,
                        help='temporal dropout in EndL')
    parser.add_argument('--endl_c_dropout', type=float, default=0.0,
                        help='channel dropout in EndL')
    parser.add_argument('--endl_embed_dropout', type=float, default=0.1,
                        help='embedding dropout in EndL')
    parser.add_argument('--endl_head_dropout', type=float, default=0.0,
                        help='head dropout in EndL')
    # WeightMLP hidden dim (maps to configs.dff in ExDM.Model)
    parser.add_argument('--dff', type=int, default=64,
                        help='hidden dim for WeightMLP temporal MLP')
    # ExoDiff hyper-parameters for ExoDiff branch
    parser.add_argument('--exodiff_steps', type=int, default=100,
                        help='number of diffusion steps in ExoDiffExogenous')
    parser.add_argument('--exodiff_hidden_dim', type=int, default=128,
                        help='hidden dim in ExoDiffExogenous denoiser MLP')
    # Inverse-loss normalization hyper-parameters
    parser.add_argument('--invnorm_p', type=float, default=1.0,
                        help='power p in inverse-loss normalization weight')
    parser.add_argument('--invnorm_sigma', type=float, default=0.1,
                        help='sigma in inverse-loss normalization weight')
    parser.add_argument('--mse_weight', type=float, default=0.5,
                        help='MSE weight when --loss is mix/mse_mae')
    parser.add_argument('--mae_weight', type=float, default=-1.0,
                        help='MAE weight when --loss is mix/mse_mae; negative means 1-mse_weight')
    parser.add_argument('--direct_weight', type=float, default=0.8,
                        help='initial/static weight of the direct endogenous forecasting head')
    parser.add_argument('--direct_dropout', type=float, default=0.05,
                        help='dropout for the direct endogenous forecasting head')
    parser.add_argument('--direct_loss_weight', type=float, default=0.2,
                        help='auxiliary loss weight for the direct endogenous forecasting head')
    parser.add_argument('--plot_exog_idx', type=int, default=0,
                        help='exogenous channel index plotted in test_results for ExDM')

    # OPTIMIZATION
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=3, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    if args.model == 'ExDM_multi_endogenous':
        if args.data not in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            parser.error('ExDM_multi_endogenous currently supports ETT datasets')
        if args.do_predict:
            parser.error('This entry supports held-out ETT forecasting, not --do_predict')
        args.target_cols = args.target_cols or ['OT', 'HUFL', 'MUFL']
        with open(os.path.join(args.root_path, args.data_path), newline='', encoding='utf-8-sig') as f:
            header = next(csv.reader(f))
        columns = header[1:]
        if (header[0] != 'date' or len(set(columns)) != len(columns)
                or len(set(args.target_cols)) != len(args.target_cols)
                or not 2 <= len(args.target_cols) < len(columns)
                or any(c not in columns for c in args.target_cols)):
            parser.error('ETT requires date first and at least two unique target columns forming a proper subset')
        if args.enc_in != len(columns) or args.output_dim != len(args.target_cols):
            parser.error('enc_in must match CSV variables; output_dim must match target_cols')
        args.model_id += '_targets_' + '_'.join(args.target_cols)
        print('Endogenous:', args.target_cols)
        print('Exogenous:', [c for c in columns if c not in args.target_cols])
    elif args.target_cols is not None:
        parser.error('--target_cols requires --model ExDM_multi_endogenous')

    if args.output_dim == 1 and args.features != 'MS':
        raise ValueError("单内生变量(output_dim=1)时，features 必须为 'MS'")
    if args.output_dim > 1 and args.features != 'M':
        raise ValueError("多内生变量(output_dim>1)时，features 必须为 'M'")

    if args.enc_in != args.input_dim + args.output_dim:
        raise ValueError(
            f"enc_in must equal input_dim+output_dim, got enc_in={args.enc_in}, "
            f"input_dim+output_dim={args.input_dim + args.output_dim}"
        )

    if args.feature_num != args.enc_in:
        raise ValueError(
            f"feature_num must equal enc_in, got feature_num={args.feature_num}, enc_in={args.enc_in}"
        )

    if args.fused_exogenous != args.input_dim:
        raise ValueError(
            f"fused_exogenous must equal input_dim, got fused_exogenous={args.fused_exogenous}, input_dim={args.input_dim}"
        )
    if args.fused_endogenous != args.output_dim:
        raise ValueError(
            f"fused_endogenous must equal output_dim, got fused_endogenous={args.fused_endogenous}, output_dim={args.output_dim}"
        )

    if args.interpdn_patch_len <= 0:
        raise ValueError(f"interpdn_patch_len must be positive, got {args.interpdn_patch_len}")
    if args.interpdn_stride <= 0:
        raise ValueError(f"interpdn_stride must be positive, got {args.interpdn_stride}")
    if args.interpdn_support_size <= 1:
        raise ValueError(f"interpdn_support_size must be > 1, got {args.interpdn_support_size}")
    if args.interpdn_patch_len > args.seq_len:
        raise ValueError(
            f"interpdn_patch_len must be <= seq_len, got interpdn_patch_len={args.interpdn_patch_len}, seq_len={args.seq_len}"
        )
    if not (0 < args.sample_alpha <= 1):
        raise ValueError(f"sample_alpha must be in (0, 1], got {args.sample_alpha}")
    if not (0 < args.sample_p < 1):
        raise ValueError(f"sample_p must be in (0, 1), got {args.sample_p}")
    if args.endl_t_ff is None:
        args.endl_t_ff = args.d_model
    if args.endl_c_ff is None:
        args.endl_c_ff = args.d_model
    if args.endl_t_ff <= 0:
        raise ValueError(f"endl_t_ff must be positive, got {args.endl_t_ff}")
    if args.endl_c_ff <= 0:
        raise ValueError(f"endl_c_ff must be positive, got {args.endl_c_ff}")
    if args.endl_t_dropout < 0 or args.endl_c_dropout < 0:
        raise ValueError(
            f"endl dropout must be >= 0, got t={args.endl_t_dropout}, c={args.endl_c_dropout}"
        )
    if args.endl_embed_dropout < 0 or args.endl_head_dropout < 0:
        raise ValueError(
            f"endl embed/head dropout must be >= 0, got embed={args.endl_embed_dropout}, head={args.endl_head_dropout}"
        )
    if args.dff <= 0:
        raise ValueError(f"dff must be positive, got {args.dff}")
    if args.invnorm_p <= 0:
        raise ValueError(f"invnorm_p must be positive, got {args.invnorm_p}")
    if args.invnorm_sigma <= 0:
        raise ValueError(f"invnorm_sigma must be positive, got {args.invnorm_sigma}")
    if not (0.0 <= args.direct_weight <= 1.0):
        raise ValueError(f"direct_weight must be in [0, 1], got {args.direct_weight}")
    if args.direct_dropout < 0:
        raise ValueError(f"direct_dropout must be >= 0, got {args.direct_dropout}")
    if args.direct_loss_weight < 0:
        raise ValueError(f"direct_loss_weight must be >= 0, got {args.direct_loss_weight}")
    if args.plot_exog_idx < 0 or args.plot_exog_idx >= args.input_dim:
        raise ValueError(
            f"plot_exog_idx must be in [0, input_dim), got plot_exog_idx={args.plot_exog_idx}, input_dim={args.input_dim}"
        )
    if args.model != 'ExDM_ablation' and args.ablation_mode != 'full':
        raise ValueError("--ablation_mode is only used with --model ExDM_ablation")
    if args.model == 'ExDM_ablation' and args.ablation_mode == 'full':
        raise ValueError("--model ExDM_ablation requires --ablation_mode in [endl_only, direct_endl, direct_exodiff]")

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.model_id_name=args.data_path[:-4]


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)


    Exp = Exp_Main


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dr{}_cin{}_rin{}_res{}_dst{}_dconv{}_efact{}'.format(
                args.model_id,
                args.model,
                args.ablation_mode if args.model == 'ExDM_ablation' else 'full',
                args.model_id_name,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.dropout,
                args.ch_ind,
                args.revin,
                args.residual,
                args.d_state,
                args.dconv,
                args.e_fact)

            exp = Exp(args)  # set experiments这个的意思应该是，我现在有了一个Exp_Main类的实例，这个实例包含了训练、测试等方法，接下来我就可以调用这些方法来进行训练和测试了。就是我现在要调用这个实验，就是通用的exp。
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dr{}_cin{}_rin{}_res{}_dst{}_dconv{}_efact{}'.format(
                args.model_id,
                args.model,
                args.ablation_mode if args.model == 'ExDM_ablation' else 'full',
                args.model_id_name,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.dropout,
                args.ch_ind,
                args.revin,
                args.residual,
                args.d_state,
                args.dconv,
                args.e_fact)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        
