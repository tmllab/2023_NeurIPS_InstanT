# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import InstantThresholdingHook, forward_loss, class_forward_loss, scale_t, error, gt_InstantThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from .T_estimator import ResNet18_F, ResNet18_N, ResNet34, ResNet50, ResNet18
import time
import datetime

def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

@ALGORITHMS.register('instant')
class InstanT(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(p_cutoff=args.p_cutoff, T=args.T, hard_label=args.hard_label, ema_p=args.ema_p)
        self.warm_up_it = args.warm_up_it
        self.total_logit_confusion = torch.zeros((self.num_classes,self.num_classes))
        self.total_logit_count = torch.zeros(self.num_classes)
        self.estimation = args.estimation
        self.scale = args.scale
        if self.estimation == "instance":
            self.T_estimator = ResNet18(self.num_classes*self.num_classes,scale=self.scale).cuda(self.args.gpu)
            self.T_optimizer = torch.optim.SGD(self.T_estimator.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9)
            self.sum = 0
        elif self.estimation == "class":
            self.vol_T = scale_t(self.gpu, self.num_classes, scale=self.scale)
            self.optimizer_vol_T = torch.optim.SGD(self.vol_T.parameters(), lr=0.01, weight_decay=0, momentum=0.9)
        else:
            raise Exception("Unsupported estimator.")
        # for evaluate T
        self.confmat = torch.zeros((self.num_classes, self.num_classes))
        self.num_calls = 0
        # end
    def init(self, p_cutoff, T, hard_label=True, ema_p=0.999):
        self.p_cutoff = p_cutoff
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p


    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='model'), 
            "DistAlignHook")
        self.register_hook(gt_InstantThresholdingHook(num_classes=self.num_classes), "MaskingHook")
        super().set_hooks()


    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}
            

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            # y_ulb = self.dataset_dict['train_ulb'].__sample__(idx_ulb.cpu())[-1]
            
            probs_x_lb = self.compute_prob(logits_x_lb.detach())
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_lb=probs_x_lb, logits_x_ulb=probs_x_ulb_w, x_ulb_w=x_ulb_w, softmax_x_lb=False, softmax_x_ulb=False)
            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)
            bool_mask = torch.gt(mask, 0)

            # if self.it % self.num_eval_iter == 0:
            #     if self.estimation == "instance":
            #         print(self.T_estimator(x_ulb_w).mean(dim=0))
            #     else:
            #         print(self.vol_T())
            # calculate loss
            if self.it < self.warm_up_it:
                unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)
            # forward correction starts
            else:
                # instance-dependent case
                if self.estimation == "instance":
                    unsup_loss = forward_loss(logits_x_ulb_s, pseudo_label, self.T_estimator(x_ulb_w), mask=mask)
                else:
                    t = self.vol_T()
                    unsup_loss = class_forward_loss(logits_x_ulb_s, pseudo_label, t, mask=mask)
                
            total_loss = sup_loss + self.lambda_u * unsup_loss
        
        out_dict = self.process_out_dict(loss=total_loss, xu_loss=unsup_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        self.out_dict = out_dict
        if self.estimation == "instance":
            self.InstanT_update()
        else:
            self.T_update()
        return out_dict, log_dict


    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint
        
    def eval_T(self, true, pred):
        return torch.abs(true - pred).mean()

    def estimate_ncp(self, x_lb, y_lb):
        out = self.model(x_lb)
        ncp = F.softmax(out['logits'],dim=-1)
        return ncp, ncp.size()[0]
    def T_update(self):
        self.optimizer_vol_T.zero_grad()
        unsup_loss = self.out_dict['xu_loss']
        unsup_loss.backward(retain_graph=True)
        self.optimizer_vol_T.step()
    def InstanT_update(self):
        self.T_optimizer.zero_grad()
        unsup_loss = self.out_dict['xu_loss']
        unsup_loss.backward(retain_graph=True)
        self.T_optimizer.step()

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
