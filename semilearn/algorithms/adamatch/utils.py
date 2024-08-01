# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
from semilearn.algorithms.hooks import MaskingHook

class AdaMatchThresholdingHook(MaskingHook):
    """
    Relative Confidence Thresholding in AdaMatch
    """

    @torch.no_grad()
    def masking(self, algorithm, logits_x_lb, logits_x_ulb, softmax_x_lb=True, softmax_x_ulb=True,  *args, **kwargs):
        if softmax_x_ulb:
            # probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        if softmax_x_lb:
            # probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_lb = algorithm.compute_prob(logits_x_lb.detach())
        else:
            # logits is already probs
            probs_x_lb = logits_x_lb.detach()

        max_probs, _ = probs_x_lb.max(dim=-1)
        p_cutoff = max_probs.mean() * algorithm.p_cutoff
        max_probs, _ = probs_x_ulb.max(dim=-1)
        mask = max_probs.ge(p_cutoff).to(max_probs.dtype)
        return mask
        
        
class InstantThresholdingHook(MaskingHook):
    """
    Instance-dependnet Thresholding
    """

    @torch.no_grad()
    def masking(self, algorithm, logits_x_lb, logits_x_ulb, x_ulb_w, softmax_x_lb=True, softmax_x_ulb=True,  *args, **kwargs):
        if softmax_x_ulb:
            # probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        if softmax_x_lb:
            # probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_lb = algorithm.compute_prob(logits_x_lb.detach())
        else:
            # logits is already probs
            probs_x_lb = logits_x_lb.detach()

        max_probs, _ = probs_x_lb.max(dim=-1)
        p_cutoff = max_probs.mean() * algorithm.p_cutoff
        max_probs, _ = probs_x_ulb.max(dim=-1)
        probs_x_ulb = torch.softmax(probs_x_ulb / algorithm.T, dim=-1)
        p_label = probs_x_ulb.argmax(dim=-1).detach().cpu()
        if algorithm.epoch < 10:
            T = algorithm.class_T
            
            thresholds = torch.zeros(probs_x_ulb.size()[0])
            thresholds += T[p_label,p_label].squeeze() * torch.topk(probs_x_ulb.detach().cpu(), 2, dim=-1)[0][:,-1]
            # j_vectors = torch.t(T)[p_label] # bs * n_class
            # j_vectors = torch.softmax(torch.t(T)[p_label] / algorithm.T, dim=-1)
            j_vectors = torch.t(T)[p_label]
            # print(j_vectors)
            j_vectors[:,p_label] = 0
        
            thresholds += (j_vectors.squeeze() * probs_x_ulb.detach().cpu().squeeze()).sum(dim=-1)
            # thresholds *= 50
            thresholds += p_cutoff.detach().cpu()
        else:
            batch_T = torch.softmax(algorithm.estimator(x_ulb_w),dim=-1).detach().cpu()
            thresholds = torch.zeros(probs_x_ulb.size()[0])
            indexes = torch.arange(batch_T.size()[0])
            thresholds += batch_T[indexes,p_label,p_label].squeeze() * torch.topk(probs_x_ulb.detach().cpu(), 2, dim=-1)[0][:,-1]
            j_vectors = batch_T[indexes,:,p_label]
            j_vectors[indexes,p_label] = 0
            thresholds += (j_vectors.squeeze() * probs_x_ulb.detach().cpu().squeeze()).sum(dim=-1)
            thresholds += p_cutoff.detach().cpu()
        thresholds = torch.clip(thresholds,0.0,0.99)
        mask = max_probs.ge(thresholds.cuda(algorithm.gpu)).to(max_probs.dtype)
        return mask
