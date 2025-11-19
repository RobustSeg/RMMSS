import torch
import torch.nn as nn
import torch.nn.functional as F


# class CriterionIFV(nn.Module):
#     def __init__(self, classes):
#         super(CriterionIFV, self).__init__()
#         self.num_classes = classes
#
#     def forward(self, feat_S, feat_T, labels):
#         feat_T = feat_T.detach()
#         size_f = feat_S.shape[2:]
#         labels = labels.unsqueeze(1)
#
#         labels_resized = F.interpolate(labels.float(), size=size_f, mode='nearest').long()
#         # print("label:", labels_resized.shape)
#
#         center_feat_S = feat_S.clone()
#         center_feat_T = feat_T.clone()
#         # print(center_feat_S.shape)
#
#         # for i in range(self.num_classes):
#         #     mask = (labels_resized == i)
#         #     print("mask shape:",mask.shape)
#         #     if mask.sum() == 0:
#         #         print("no mask feat:",center_feat_S.shape)
#         #         mask_feat_S = torch.zeros_like(center_feat_S[0, :, 0, 0])
#         #         print("empty feat:", mask_feat_S.shape)
#         #         mask_feat_T = torch.zeros_like(center_feat_T[0, :, 0, 0])
#         #     else:
#         #         # view用来将向量拉成一维，nonzero用来找到True的部分，返回（N，1），squeeze(1)返回(N,)
#         #         mask_idx = mask.view(-1).nonzero().squeeze(1)
#         #         print("mask_idx shape:",mask_idx.shape)
#         #         mask_feat_S = feat_S.flatten(2)[:, :, mask_idx].mean(dim=2)
#         #         print("mask feat:", mask_feat_S.shape)
#         #         mask_feat_T = feat_T.flatten(2)[:, :, mask_idx].mean(dim=2)
#         #
#         #     # Expand mask_feat_S and mask_feat_T to [1, C, 1, 1]
#         #     mask_feat_S = mask_feat_S.unsqueeze(-1).unsqueeze(-1)
#         #     mask_feat_T = mask_feat_T.unsqueeze(-1).unsqueeze(-1)
#         #
#         #     # Center projection
#         #     center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * (
#         #                 (mask_feat_S * feat_S).sum(dim=(2, 3), keepdim=True) / (
#         #                     mask_feat_S.sum(dim=(2, 3), keepdim=True) + 1e-6))
#         #     center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * (
#         #                 (mask_feat_T * feat_T).sum(dim=(2, 3), keepdim=True) / (
#         #                     mask_feat_T.sum(dim=(2, 3), keepdim=True) + 1e-6))
#         #
#         # # Cosine similarity along C dimension
#         # cos = nn.CosineSimilarity(dim=1)
#         # pcsim_feat_S = cos(feat_S, center_feat_S)
#         # pcsim_feat_T = cos(feat_T, center_feat_T)
#         #
#         # # MSE Loss
#         # mse = nn.MSELoss()
#         # loss = mse(pcsim_feat_S, pcsim_feat_T)
#         # return loss
#         for i in range(self.num_classes):
#           mask_feat_S = (labels_resized == i).float()
#           mask_feat_T = (labels_resized == i).float()
#           center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * ((mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
#           center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * ((mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
#
#         # cosinesimilarity along C
#         cos = nn.CosineSimilarity(dim=1)
#         pcsim_feat_S = cos(feat_S, center_feat_S)
#         pcsim_feat_T = cos(feat_T, center_feat_T)
#
#         # mseloss
#         mse = nn.MSELoss()
#         loss = mse(pcsim_feat_S, pcsim_feat_T)
#         return loss

class CriterionIFV(nn.Module):
    def __init__(self, classes):
        super(CriterionIFV, self).__init__()
        self.num_classes = classes

    def forward(self, feat_S, feat_T, target):
        # print(feat_S.shape,feat_T.shape, target.shape)
        # example: torch.Size([1, 32, 128, 128]) torch.Size([1, 32, 128, 128]) torch.Size([1, 512, 512])
        feat_T.detach()
        size_f = feat_S.shape[2:]
        tar_feat_S = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_S.size())
        tar_feat_T = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_T.size())
        center_feat_S = feat_S.clone()
        center_feat_T = feat_T.clone()
        for i in range(self.num_classes):
          mask_feat_S = (tar_feat_S == i).float()
          mask_feat_T = (tar_feat_T == i).float()
          center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * ((mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
          center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * ((mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)

        # cosinesimilarity along C
        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(feat_S, center_feat_S)
        print(feat_S.shape, center_feat_S.shape, pcsim_feat_S.shape)
        pcsim_feat_T = cos(feat_T, center_feat_T)

        # mseloss
        # mse = nn.MSELoss()
        # loss = mse(pcsim_feat_S, pcsim_feat_T)
        # loss_kl = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
        # feat_S_log_softmax = torch.log_softmax(pcsim_feat_S.view(1, -1),dim=-1)
        # feat_T_softmax = torch.softmax(pcsim_feat_T.view(1, -1), dim=-1)
        # loss = loss_kl(feat_S_log_softmax, feat_T_softmax).clamp(min=0)
        # return loss
        T = 2.0
        p_T = F.softmax(pcsim_feat_T / T, dim=-1)
        log_q_S = F.log_softmax(pcsim_feat_S / T, dim=-1)
        loss = F.kl_div(log_q_S, p_T, reduction='batchmean') * (T * T)

        return loss

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