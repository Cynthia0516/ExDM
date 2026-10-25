from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import ExDM
from models import ExDM_ablation
from models import ExDM_multi_endogenous
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import pandas as pd
import os
import json
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # EMA statistics of n+1 losses (n EndL branch losses + 1 WeightMLP loss)
        # used as predict-time fusion priors when future labels are unavailable.
        self._invnorm_loss_ema = None

    def _ema_ckpt_path(self, ckpt_dir):
        return os.path.join(ckpt_dir, 'invnorm_ema.pth')

    def _save_invnorm_ema(self, ckpt_dir):
        if self._invnorm_loss_ema is None:
            return
        torch.save({'invnorm_loss_ema': self._invnorm_loss_ema.detach().cpu()}, self._ema_ckpt_path(ckpt_dir))

    def _load_invnorm_ema(self, ckpt_dir):
        ema_path = self._ema_ckpt_path(ckpt_dir)
        if not os.path.exists(ema_path):
            return
        payload = torch.load(ema_path, map_location=self.device)
        ema = payload.get('invnorm_loss_ema', None) if isinstance(payload, dict) else None
        if ema is not None:
            self._invnorm_loss_ema = ema.to(self.device)

    def _unpack_batch(self, batch):
        if len(batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = batch
        elif len(batch) == 4:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_cycle = None
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
        return batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle

    def _build_model(self):
        model_dict = {
            'ExDM': ExDM,
            'ExDM_ablation': ExDM_ablation,
            'ExDM_multi_endogenous': ExDM_multi_endogenous,
        }
        model = model_dict[self.args.model].Model(self.args).float()#ExDM.Model(self.args).float()
        print(model)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _is_exdm_family(self):
        return self.args.model in ('ExDM', 'ExDM_ablation', 'ExDM_multi_endogenous')

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        loss_type = str(getattr(self.args, 'loss', 'mse')).lower()
        if loss_type in ('mse', 'l2'):
            return nn.MSELoss()
        if loss_type in ('mae', 'l1'):
            return nn.L1Loss()
        if loss_type in ('huber', 'smoothl1'):
            return nn.SmoothL1Loss()
        if loss_type in ('mix', 'mixed', 'mse_mae', 'mse+mae'):
            mse_weight = float(getattr(self.args, 'mse_weight', 0.5))
            mae_weight = float(getattr(self.args, 'mae_weight', -1.0))
            if mae_weight < 0.0:
                mae_weight = 1.0 - mse_weight
            mse_weight = max(mse_weight, 0.0)
            mae_weight = max(mae_weight, 0.0)
            total = mse_weight + mae_weight
            if total <= 0.0:
                mse_weight, mae_weight, total = 0.5, 0.5, 1.0
            mse_weight /= total
            mae_weight /= total

            def criterion(pred, target):
                mse = torch.mean((pred - target) ** 2)
                mae = torch.mean(torch.abs(pred - target))
                return mse_weight * mse + mae_weight * mae

            return criterion
        return nn.MSELoss()

    def _normalize_feature(self, x, eps=1e-6):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / (std + eps)

    def _align_time_dim_pair(self, a, b):
        """Align two tensors on time dimension (dim=1) by right-cropping to the same length."""
        if a.dim() < 2 or b.dim() < 2:
            return a, b
        if a.size(1) == b.size(1):
            return a, b
        t = min(a.size(1), b.size(1))
        return a[:, -t:, ...], b[:, -t:, ...]

    def _inverse_loss_weights(self, losses):
        # losses: [n+1]
        p = float(getattr(self.args, 'invnorm_p', 1.0))
        sigma = float(getattr(self.args, 'invnorm_sigma', 0.1))
        p = max(p, 1e-8)
        sigma = max(sigma, 1e-8)
        inv = 1.0 / (torch.pow(losses + sigma, p) + 1e-8)
        return inv / (inv.sum() + 1e-8)

    def _update_invnorm_loss_ema(self, losses):
        # losses: [n+1], detached tensor
        momentum = float(getattr(self.args, 'invnorm_ema_momentum', 0.9))
        momentum = min(max(momentum, 0.0), 0.9999)
        if self._invnorm_loss_ema is None or self._invnorm_loss_ema.numel() != losses.numel():
            self._invnorm_loss_ema = losses.detach().clone()
        else:
            self._invnorm_loss_ema = momentum * self._invnorm_loss_ema + (1.0 - momentum) * losses.detach()

    def _exdm_predict_fused_output(self, model_out):
        outputs, comp = model_out
        if comp.get('ablation_mode', None) is not None:
            return outputs[:, -self.args.pred_len:, :]
        if 'direct_output' in comp:
            return outputs[:, -self.args.pred_len:, :]

        # Predict stage has no future labels; reuse training-optimized EMA losses to build fusion weights.
        endo_list = comp.get('fused_endogenous_pred_list', None)
        weightmlp_output = comp.get('weightmlp_output', outputs)
        weightmlp_output = weightmlp_output[:, -self.args.pred_len:, :]

        if endo_list is None:
            return weightmlp_output

        endl_outputs = endo_list[:, :, -self.args.pred_len:, :]  # [n, B, H, output_dim]
        all_outputs = torch.cat([endl_outputs, weightmlp_output.unsqueeze(0)], dim=0)  # [n+1, B, H, output_dim]
        k = all_outputs.size(0)

        if self._invnorm_loss_ema is not None and self._invnorm_loss_ema.numel() == k:
            losses_for_weight = self._invnorm_loss_ema.to(all_outputs.device)
            wi = self._inverse_loss_weights(losses_for_weight).view(-1, 1, 1, 1)
        else:
            wi = torch.full((k, 1, 1, 1), 1.0 / k, device=all_outputs.device)

        return (all_outputs * wi).sum(dim=0)

    def _exdm_two_part_loss(self, model_out, batch_y, criterion):
        outputs, comp = model_out
        if comp.get('ablation_mode', None) is not None:
            return self._exdm_ablation_loss(outputs, comp, batch_y, criterion)

        fused_exogenous_pred = comp['fused_exogenous_pred'][:, -self.args.pred_len:, :]
        fused_endogenous_pred = comp['fused_endogenous_pred'][:, -self.args.pred_len:, :]
        fused_endogenous_pred_list = comp.get('fused_endogenous_pred_list', None)
        weightmlp_output = comp.get('weightmlp_output', outputs)
        weightmlp_output = weightmlp_output[:, -self.args.pred_len:, :]
        direct_output = comp.get('direct_output', outputs)
        direct_output = direct_output[:, -self.args.pred_len:, :]

        batch_y_exo = batch_y[:, -self.args.pred_len:, :self.args.input_dim].to(self.device)
        fused_exogenous_pred, batch_y_exo = self._align_time_dim_pair(fused_exogenous_pred, batch_y_exo)
        fused_exogenous_norm = self._normalize_feature(batch_y_exo).detach()
        fused_exogenous_pred_norm = self._normalize_feature(fused_exogenous_pred)

        loss1 = criterion(fused_exogenous_pred_norm, fused_exogenous_norm)

        start = self.args.input_dim
        end = start + self.args.output_dim
        if end > batch_y.shape[-1]:
            raise ValueError(
                f"batch_y channels ({batch_y.shape[-1]}) < input_dim+output_dim ({end})"
            )
        batch_y_endo = batch_y[:, -self.args.pred_len:, start:end].to(self.device)

        if fused_endogenous_pred_list is not None:
            # fused_endogenous_pred_list: [n, B, H, output_dim]
            loss2_list = []
            endl_outputs = []
            for i in range(fused_endogenous_pred_list.size(0)):
                pred_i = fused_endogenous_pred_list[i, :, -self.args.pred_len:, :]
                pred_i, target_i = self._align_time_dim_pair(pred_i, batch_y_endo)
                loss2_list.append(criterion(pred_i, target_i))
                endl_outputs.append(pred_i)
            loss2_all = torch.stack(loss2_list)
            loss2 = loss2_all.mean()
        else:
            fused_endogenous_pred, batch_y_endo_for_loss2 = self._align_time_dim_pair(fused_endogenous_pred, batch_y_endo)
            loss2 = criterion(fused_endogenous_pred, batch_y_endo_for_loss2)
            loss2_all = loss2.unsqueeze(0)
            endl_outputs = [fused_endogenous_pred]
        outputs_wmlp = weightmlp_output

        # 保留原有 WeightMLP 输出与 loss3。
        outputs_wmlp = weightmlp_output
        outputs_wmlp, batch_y_endo_for_loss3 = self._align_time_dim_pair(outputs_wmlp, batch_y_endo)
        loss3 = criterion(outputs_wmlp, batch_y_endo_for_loss3)
        direct_output, batch_y_endo_for_direct = self._align_time_dim_pair(direct_output, batch_y_endo)
        loss_direct = criterion(direct_output, batch_y_endo_for_direct)
        all_losses = torch.cat([loss2_all, loss3.unsqueeze(0)], dim=0)

        # n+1 个分支（n 个 EndL + 1 个 WeightMLP）做反比归一化加权。
        all_losses = torch.cat([loss2_all, loss3.unsqueeze(0)], dim=0)
        if self.model.training and torch.is_grad_enabled():
            self._update_invnorm_loss_ema(all_losses.detach())
        outputs, batch_y_endo_for_final = self._align_time_dim_pair(outputs, batch_y_endo)
        loss_final = criterion(outputs, batch_y_endo_for_final)
        direct_loss_weight = float(getattr(self.args, 'direct_loss_weight', 0.2))
        loss = 0.03*loss1 + 0.05*loss2 + 0.12*loss3 + loss_final + direct_loss_weight*loss_direct
        # pred_len = self.args.pred_len
        # loss1_weight = 0.05 * (pred_len / 96) ** 1.1
        # loss2_weight = 0.07
        # loss3_weight = max(0.7, 1.0 - loss1_weight - loss2_weight)
        # loss = loss1_weight * loss1 + loss2_weight * loss2 + loss3_weight * loss3
        return outputs, loss, loss1, loss2, loss3

    def _exdm_ablation_loss(self, outputs, comp, batch_y, criterion):
        zero = outputs.sum() * 0.0
        loss1 = zero
        loss2 = zero
        loss3 = zero

        start = self.args.input_dim
        end = start + self.args.output_dim
        if end > batch_y.shape[-1]:
            raise ValueError(
                f"batch_y channels ({batch_y.shape[-1]}) < input_dim+output_dim ({end})"
            )
        batch_y_endo = batch_y[:, -self.args.pred_len:, start:end].to(self.device)
        outputs, batch_y_endo_for_final = self._align_time_dim_pair(outputs, batch_y_endo)
        loss_final = criterion(outputs, batch_y_endo_for_final)
        loss = loss_final

        fused_exogenous_pred = comp.get('fused_exogenous_pred', None)
        if fused_exogenous_pred is not None:
            batch_y_exo = batch_y[:, -self.args.pred_len:, :self.args.input_dim].to(self.device)
            fused_exogenous_pred = fused_exogenous_pred[:, -self.args.pred_len:, :]
            fused_exogenous_pred, batch_y_exo = self._align_time_dim_pair(fused_exogenous_pred, batch_y_exo)
            loss1 = criterion(self._normalize_feature(fused_exogenous_pred), self._normalize_feature(batch_y_exo).detach())
            loss = loss + 0.03 * loss1

        fused_endogenous_pred = comp.get('fused_endogenous_pred', None)
        if fused_endogenous_pred is not None and comp.get('ablation_mode') == 'direct_endl':
            fused_endogenous_pred = fused_endogenous_pred[:, -self.args.pred_len:, :]
            fused_endogenous_pred, target = self._align_time_dim_pair(fused_endogenous_pred, batch_y_endo)
            loss2 = criterion(fused_endogenous_pred, target)
            loss = loss + 0.05 * loss2

        weightmlp_output = comp.get('weightmlp_output', None)
        if weightmlp_output is not None:
            weightmlp_output = weightmlp_output[:, -self.args.pred_len:, :]
            weightmlp_output, target = self._align_time_dim_pair(weightmlp_output, batch_y_endo)
            loss3 = criterion(weightmlp_output, target)
            loss = loss + 0.12 * loss3

        direct_output = comp.get('direct_output', None)
        if direct_output is not None and comp.get('ablation_mode') in ('direct_endl', 'direct_exodiff'):
            direct_output = direct_output[:, -self.args.pred_len:, :]
            direct_output, target = self._align_time_dim_pair(direct_output, batch_y_endo)
            direct_loss_weight = float(getattr(self.args, 'direct_loss_weight', 0.2))
            loss = loss + direct_loss_weight * criterion(direct_output, target)

        return outputs, loss, loss1, loss2, loss3

    def _align_outputs_targets(self, outputs, batch_y):
        outputs = outputs[:, -self.args.pred_len:, :]
        if self._is_exdm_family():
            start = self.args.input_dim
            end = start + self.args.output_dim
            if end > batch_y.shape[-1]:
                raise ValueError(
                    f"batch_y channels ({batch_y.shape[-1]}) < input_dim+output_dim ({end})"
                )
            batch_y = batch_y[:, -self.args.pred_len:, start:end]
            return outputs, batch_y

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, :, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        return outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)        
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if batch_cycle is not None:
                    batch_cycle = batch_cycle.long().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        #if 'Machine' in self.args.model:（下面那一行是ai）
                        if 'Machine' in self.args.model or self._is_exdm_family():
                            if self._is_exdm_family():
                                model_out = self.model(batch_x, batch_cycle=batch_cycle, return_components=True)
                                outputs, loss, _, _, _ = self._exdm_two_part_loss(model_out, batch_y, criterion)
                            else:
                                outputs = self.model(batch_x)
                                outputs, batch_y = self._align_outputs_targets(outputs, batch_y)
                                batch_y = batch_y.to(self.device)
                                pred = outputs.detach().cpu()
                                true = batch_y.detach().cpu()
                                loss = criterion(pred, true)
                        
                else:
                    #if 'Machine' in self.args.model:（下面那一行是ai）
                    if 'Machine' in self.args.model or self._is_exdm_family():
                        if self._is_exdm_family():
                            model_out = self.model(batch_x, batch_cycle=batch_cycle, return_components=True)
                            outputs, loss, _, _, _ = self._exdm_two_part_loss(model_out, batch_y, criterion)
                        else:
                            outputs = self.model(batch_x)
                            outputs, batch_y = self._align_outputs_targets(outputs, batch_y)
                            batch_y = batch_y.to(self.device)
                            pred = outputs.detach().cpu()
                            true = batch_y.detach().cpu()
                            loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        best_val_loss_for_ema = np.inf
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = self._unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if batch_cycle is not None:
                    batch_cycle = batch_cycle.long().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Machine' in self.args.model or self._is_exdm_family():
                            if self._is_exdm_family():
                                model_out = self.model(batch_x, batch_cycle=batch_cycle, return_components=True)
                                outputs, loss, loss1, loss2, loss3 = self._exdm_two_part_loss(model_out, batch_y, criterion)
                            else:
                                outputs = self.model(batch_x)
                                outputs, batch_y = self._align_outputs_targets(outputs, batch_y)
                                batch_y = batch_y.to(self.device)
                                loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    #if 'Machine' in self.args.model or self._is_exdm_family():（下面那2行是ai）
                        #outputs = self.model(batch_x)
                    if 'Machine' in self.args.model or self._is_exdm_family():
                        if self._is_exdm_family():
                            model_out = self.model(batch_x, batch_cycle=batch_cycle, return_components=True)
                            outputs, loss, loss1, loss2, loss3 = self._exdm_two_part_loss(model_out, batch_y, criterion)
                        else:
                            outputs = self.model(batch_x)
                            # print(outputs.shape,batch_y.shape)
                            outputs, batch_y = self._align_outputs_targets(outputs, batch_y)
                            batch_y = batch_y.to(self.device)
                            loss = criterion(outputs, batch_y)
                    
                    train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    if self._is_exdm_family():
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} | loss1: {3:.7f} | loss2: {4:.7f} | loss3: {5:.7f}".format(
                            i + 1, epoch + 1, loss.item(), loss1.item(), loss2.item(), loss3.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if vali_loss < best_val_loss_for_ema:
                best_val_loss_for_ema = vali_loss
                self._save_invnorm_ema(path)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self._load_invnorm_ema(path)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        criterion = self._select_criterion()
        
        if test:
            print('loading model')
            ckpt_dir = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))
            self._load_invnorm_ema(ckpt_dir)

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if batch_cycle is not None:
                    batch_cycle = batch_cycle.long().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        #if 'Machine' in self.args.model:（下面那一行是ai）
                        if 'Machine' in self.args.model or self._is_exdm_family():
                            if self._is_exdm_family():
                                model_out = self.model(batch_x, batch_cycle=batch_cycle, return_components=True)
                                outputs = self._exdm_predict_fused_output(model_out)
                            else:
                                outputs = self.model(batch_x, batch_cycle=batch_cycle)
                        
                else:
                    #if 'Machine' in self.args.model:(下面那一行是ai)
                    if 'Machine' in self.args.model or self._is_exdm_family():
                            if self._is_exdm_family():
                                model_out = self.model(batch_x, batch_cycle=batch_cycle, return_components=True)
                                outputs = self._exdm_predict_fused_output(model_out)
                            else:
                                outputs = self.model(batch_x, batch_cycle=batch_cycle)

                if self._is_exdm_family():
                    start = self.args.input_dim
                    end = start + self.args.output_dim
                    if end > batch_y.shape[-1]:
                        raise ValueError(
                            f"batch_y channels ({batch_y.shape[-1]}) < input_dim+output_dim ({end})"
                        )
                    batch_y = batch_y[:, -self.args.pred_len:, start:end]
                else:
                    outputs, batch_y = self._align_outputs_targets(outputs, batch_y)
                batch_y = batch_y.to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    prd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, prd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()
        temp_df = pd.DataFrame()
        temp_df['Seed']=[self.args.random_seed]
        temp_df['Model']=[self.args.model]
        ablation_mode = getattr(self.args, 'ablation_mode', 'full') if self.args.model == 'ExDM_ablation' else 'full'
        temp_df['ablation_mode']=[ablation_mode]
        temp_df['seq_len']=[self.args.seq_len]
        temp_df['label_len']=[self.args.label_len]
        temp_df['pred_len']=[self.args.pred_len]
        #temp_df['n1']=[self.args.n1]
        #temp_df['n2']=[self.args.n2]
        #temp_df['dropout']=[self.args.dropout]
        temp_df['train_epochs']=[self.args.train_epochs]
        temp_df['batch']=[self.args.batch_size]
        temp_df['patience']=[self.args.patience]
        temp_df['LR']=[self.args.learning_rate]
        temp_df['dropout']=[self.args.dropout]
        temp_df['ch_ind']=[self.args.ch_ind]
        temp_df['revin']=[self.args.revin]
        temp_df['e_fact']=[self.args.e_fact]
        temp_df['dconv']=[self.args.dconv]

        temp_df['MSE']=[mse]
        temp_df['MAE']=[mae]
        temp_df['residual']=[self.args.residual]
        temp_df['d_state']=[self.args.d_state]

        temp_df['checkpoint_path']=[setting]
        os.makedirs('./csv_results', exist_ok=True)
        if self.args.model == 'ExDM_multi_endogenous':
            temp_df['target_cols'] = [','.join(self.args.target_cols)]
            for i, name in enumerate(self.args.target_cols):
                error = preds[:, :, i] - trues[:, :, i]
                temp_df[name + '_MSE'] = [float(np.mean(error ** 2))]
                temp_df[name + '_MAE'] = [float(np.mean(np.abs(error)))]

        if not os.path.exists('./csv_results/'+'result_'+self.args.data_path):
            temp_df.to_csv('./csv_results/'+'result_'+self.args.data_path, index=False)
        else:
            result_df=pd.read_csv('./csv_results/'+'result_'+self.args.data_path)
            result_df = pd.concat([result_df,temp_df],ignore_index=True)
            result_df.to_csv('./csv_results/'+'result_'+self.args.data_path, index=False)

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)
        if self.args.model == 'ExDM_multi_endogenous':
            per_target = {name: {'mse': float(temp_df[name + '_MSE'].iloc[0]),
                                 'mae': float(temp_df[name + '_MAE'].iloc[0])}
                          for name in self.args.target_cols}
            with open(folder_path + 'metrics.json', 'w', encoding='utf-8') as f:
                json.dump({'target_cols': self.args.target_cols,
                           'scale': 'training-set standardized',
                           'mse': float(mse), 'mae': float(mae),
                           'per_target': per_target}, f, indent=2)
            print('Endogenous metrics:', per_target)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            self._load_invnorm_ema(path)

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(pred_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if batch_cycle is not None:
                    batch_cycle = batch_cycle.long().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        #if 'Machine' in self.args.model:(下面那一行是ai)
                        if 'Machine' in self.args.model or self._is_exdm_family():
                            if self._is_exdm_family():
                                model_out = self.model(batch_x, batch_cycle=batch_cycle, return_components=True)
                                outputs = self._exdm_predict_fused_output(model_out)
                            else:
                                outputs = self.model(batch_x, batch_cycle=batch_cycle)
                       
                else:
                    #if 'Machine' in self.args.model:（下面那一行是ai）
                    if 'Machine' in self.args.model or self._is_exdm_family():
                        if self._is_exdm_family():
                            model_out = self.model(batch_x, batch_cycle=batch_cycle, return_components=True)
                            outputs = self._exdm_predict_fused_output(model_out)
                        else:
                            outputs = self.model(batch_x, batch_cycle=batch_cycle)
                    
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
