import torch
import torch.nn as nn
from mmkd.IFV import CriterionIFV
# from semseg.models.segformer.seg_block_UMDt import Seg as Seg_s
# from semseg.models.segformer.seg_block_select_UMD import Seg


def IFVMD(x_all: list, x_all_t: list, lbl: torch.Tensor, num_classes:int):
    loss_ifv = 0.0
    stage = len(x_all)
    B = x_all[0].shape[0]
    criterion = CriterionIFV(classes=num_classes)
    for i in range(stage):
        # print(batch_label.shape)
        for j in range(B):
            batch_label = lbl[j].unsqueeze(0)
            # x_all[i][j]维度为x_all torch.Size([32, 256, 256])
            # 为了迎合prototype类中的batch需要，此处unsqueeze(0)
            x_all_s_feature = x_all[i][j].unsqueeze(0)
            x_all_t_feature = x_all_t[i][j].unsqueeze(0)

            loss = criterion(x_all_s_feature, x_all_t_feature, batch_label)

            # print("prototype",x_all_s_softmax_prototype.sjianhape)

            loss_ifv += loss


    return loss_ifv / B

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
# loss = IFVMD(index, ms_feat, ms_feat_s, lbl, model.num_classes)
# print(0)