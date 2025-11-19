import torch
import torch.nn as nn
import random
from mmkd.regularization import Perturbation, Regularization, RegParameters
# from semseg.models.segformer.seg_block_UMDt import Seg as Seg_s
# from semseg.models.segformer.seg_block_select_UMD import Seg


def MB(sample: list, reg_params:RegParameters, model, logits):
    loss = 0.0
    loss_kl = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
    # print(logits.shape)
    expand_logits = Perturbation.get_expanded_logits(logits, reg_params.n_samples,logits_flg=False)
    inf_sample = [Perturbation.perturb_tensor(x, reg_params.n_samples) for x in sample]
    #inf_sample = [x for x in sample]
    #
    with torch.no_grad():
        inf_output, _, _ = model(inf_sample)

    # # print("ddd",expand_logits.shape)
    #
    inf_output_log_softmax = torch.log_softmax(inf_output, dim=1)
    expand_logits_softmax = torch.softmax(expand_logits, dim=1)
    #
    inf_loss = loss_kl(inf_output_log_softmax, expand_logits_softmax).clamp(min=0)
    #
    # gradients = torch.autograd.grad(inf_loss, inf_sample, create_graph=True)
    # grads = [Regularization.get_batch_norm(gradients[k], loss=inf_loss,
    #                                        estimation=reg_params.estimation) for k in range(2)]
    #
    # inf_scores = torch.stack(grads)
    # reg_term = Regularization.get_regularization_term(inf_scores, norm=reg_params.norm,
    #                                                   optim_method=reg_params.optim_method)
    #
    # loss = loss +  reg_params.lambda_ * reg_term
    loss += inf_loss


    return loss

#
# model = Seg("mit_b0", num_classes=25, pretrained=True)
# model_s = Seg_s("mit_b0", num_classes=25, pretrained=True)
#
# sample = [torch.zeros(2, 3, 1024, 1024), torch.ones(2, 3, 1024, 1024), torch.ones(2, 3, 1024, 1024),
#           torch.ones(2, 3, 1024, 1024)]
# lbl = torch.zeros(2, 1024, 1024)
#
# logits, index, ms_feat = model(sample)
# with torch.no_grad():
#     logits_s, ms_feat_s = model_s(sample)
# loss = PUMD(index, ms_feat, ms_feat_s, lbl, model.num_classes)
# print(0)