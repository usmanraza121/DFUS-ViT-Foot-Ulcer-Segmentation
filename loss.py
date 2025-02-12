import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision
import torchvision.transforms as F

from IPython.display import display
def structure_loss(pred, mask):
    #print('pred',pred.shape)
    #print('mask',mask.shape)
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=11, stride=1, padding=5) - mask)
    # print('weit',weit.shape)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()


#https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained
'''  focal loss '''
#https://blog.paperspace.com/pytorch-loss-functions/
#https://github.com/doiken23/pytorch_toolbox/blob/master/focalloss2d.py
class FocalLoss2d(nn.Module):

    def __init__(self, gamma=5, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.binary_cross_entropy_with_logits(input, target, reduce='none')
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

import torch
import torch.nn.functional as F


def hybrid_e_loss(pred, mask):
    """
    Introduction: 
        Hybrid Eloss
    Paramaters:
        :param pred:
        :param mask:
        :return:
    Usage:
        loss = hybrid_e_loss(prediction_map, gt_mask)
        loss.backward()
    """
    # adaptive weighting masks
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    # weighted binary cross entropy loss function
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = ((weit * wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)

    # weighted e loss function
    pred = torch.sigmoid(pred)
    mpred = pred.mean(dim=(2, 3)).view(pred.shape[0], pred.shape[1], 1, 1).repeat(1, 1, pred.shape[2], pred.shape[3])
    phiFM = pred - mpred

    mmask = mask.mean(dim=(2, 3)).view(mask.shape[0], mask.shape[1], 1, 1).repeat(1, 1, mask.shape[2], mask.shape[3])
    phiGT = mask - mmask

    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))

    # weighted iou loss function
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)

    return (wbce + eloss + wiou).mean()


''' KL Divergence loss'''

class KLD(nn.Module):
    def __init__(self, reduction="mean"):
        super(KLD, self).__init__()
        self.reduction = reduction
        self.kld = nn.KLDivLoss(reduction="none", log_target=False)

    def forward(self, output, target):
        assert len(output.shape) == 4, "Must be 4-dimensional output"
        output = torch.log_softmax(output, dim=1)
        loss = self.kld(output, target)
        loss = torch.sum(loss, dim=1)
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum": return loss.sum()
        return loss
"""Focal Loss"""
class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0.5, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

"""BCEFocalLoss"""
class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

if __name__ == "__main__":
    focal_loss = BCEFocalLoss()
    x = torch.nn.Sigmoid()(torch.randn((3, 3)))
    y = torch.nn.Sigmoid()(torch.randn((3, 3)))
    loss = focal_loss(x, y)
    print(loss)


"""Weighted Focol Loass"""
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"    
    def __init__(self, alpha=.25, gamma=2):
            super(WeightedFocalLoss, self).__init__()        
            self.alpha = torch.tensor([alpha, 1-alpha]).cuda()        
            self.gamma = gamma
            
    def forward(self, inputs, targets):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')        
            targets = targets.type(torch.long)        
            at = self.alpha.gather(0, targets.data.view(-1))        
            pt = torch.exp(-BCE_loss)        
            F_loss = at*(1-pt)**self.gamma * BCE_loss        
            return F_loss.mean()


"""BCE loss"""

class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss


""" Deep Supervision Loss"""


def DeepSupervisionLoss(pred, gt):
    d0, d1, d2, d3, d4 = pred[0:]

    criterion = BceDiceLoss()

    loss0 = criterion(d0, gt)
    gt = torch.nn.functional.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss1 = criterion(d1, gt)
    gt = torch.nn.functional.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss2 = criterion(d2, gt)
    gt = torch.nn.functional.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss3 = criterion(d3, gt)
    gt = torch.nn.functional.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss4 = criterion(d4, gt)

    return loss0 + loss1 + loss2 + loss3 + loss4



