# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch 
import torch.nn as nn

from torch.nn import functional as F


def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)

def forward_loss(logits, targets, batch_T, reduction='none'):
    if logits.shape == targets.shape:
        softmax_pred = torch.softmax(logits, dim=-1)
        corr_pred = torch.bmm(logits.unsqueeze(1), batch_T).squeeze()
        log_corr_pred = torch.log(corr_pred + 1e-12)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return cross_loss
    else:
        softmax_pred = torch.softmax(logits, dim=-1)
        corr_pred = torch.bmm(logits.unsqueeze(1), batch_T).squeeze()
        log_corr_pred = torch.log(corr_pred + 1e-12)
        return F.nll_loss(log_corr_pred, targets, reduction=reduction)

class CELoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def forward(self, logits, targets, reduction='none'):
        return ce_loss(logits, targets, reduction)

class forward_correction(nn.Module):
    def forward(self, logits, targets, batch_T, reduction='none'):
        return forward_loss(logits, targets, batch_T)