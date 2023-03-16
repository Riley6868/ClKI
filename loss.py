import numpy as np
import torch
import torch.nn as nn
import logging
# 支持多分类和二分类
import torch.nn.functional as F
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)  # 这里看情况选择，如果之前softmax了，后续就不用了

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Loss_func(nn.Module):

    def __init__(self, args,device):
        super(Loss_func, self).__init__()
        #self.criterion = nn.CrossEntropyLoss().to(device)
        self.criterion = FocalLoss(2)
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.middle1_losses = AverageMeter()
        self.middle2_losses = AverageMeter()
        self.middle3_losses = AverageMeter()

        # self.middle4_losses = AverageMeter()
        # self.middle5_losses = AverageMeter()


        self.losses1_kd = AverageMeter()
        self.losses2_kd = AverageMeter()
        self.losses3_kd = AverageMeter()

        # self.losses4_kd = AverageMeter()
        # self.losses5_kd = AverageMeter()


        self.feature_losses_1 = AverageMeter()
        self.feature_losses_2 = AverageMeter()
        self.feature_losses_3 = AverageMeter()
        # self.feature_losses_4 = AverageMeter()

        self.total_losses = AverageMeter()
        self.middle1_top1 = AverageMeter()
        self.middle2_top1 = AverageMeter()
        self.middle3_top1 = AverageMeter()

        # self.middle4_top1 = AverageMeter()
        # self.middle5_top1 = AverageMeter()

        self.args = args

    def kd_loss_function(self,output, target_output):
        """Compute kd loss"""
        """
        para: output: middle ouptput logits.
        para: target_output: final output has divided by temperature and softmax.
        """

        output = output / self.args.temperature
        output_log_softmax = torch.log_softmax(output, dim=1)
        loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
        return loss_kd

    def feature_loss_function(self,fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss).sum()


    def forward(self, out_pred,out_fea,out_x,target,step,writer,flag):

        #loss_0 = self.criterion(out_pred[0], target)

        loss = self.criterion(out_pred[-1], target)
        #loss = self.criterion(out_x,target)
        self.losses.update(loss.item(), out_pred[-1].shape[0])
        #
        middle1_loss = self.criterion(out_pred[0], target)
        self.middle1_losses.update(middle1_loss.item(), out_pred[-1].shape[0])
        #
        middle2_loss = self.criterion(out_pred[1], target)
        self.middle2_losses.update(middle2_loss.item(), out_pred[-1].shape[0])

        middle3_loss = self.criterion(out_pred[2], target)
        self.middle3_losses.update(middle3_loss.item(), out_pred[-1].shape[0])
        # #
        # middle4_loss = self.criterion(out_pred[3], target)
        # self.middle4_losses.update(middle4_loss.item(), out_pred[-1].shape[0])
        # #
        # middle5_loss = self.criterion(out_pred[4], target)
        # self.middle5_losses.update(middle5_loss.item(), out_pred[-1].shape[0])

        temp4 = out_pred[-1] / self.args.temperature
        temp4 = torch.softmax(temp4, dim=1)
        # #
        loss1by4 = self.kd_loss_function(out_pred[0], temp4) * (self.args.temperature ** 2)
        self.losses1_kd.update(loss1by4, out_pred[-1].shape[0])
        #
        loss2by4 = self.kd_loss_function(out_pred[1], temp4) * (self.args.temperature ** 2)
        self.losses2_kd.update(loss2by4, out_pred[-1].shape[0])

        loss3by4 = self.kd_loss_function(out_pred[2], temp4) * (self.args.temperature ** 2)
        self.losses3_kd.update(loss3by4, out_pred[-1].shape[0])
        #
        # loss4by4 = self.kd_loss_function(out_pred[3], temp4) * (self.args.temperature ** 2)
        # self.losses4_kd.update(loss4by4, out_pred[-1].shape[0])
        # #
        # loss5by4 = self.kd_loss_function(out_pred[4], temp4) * (self.args.temperature ** 2)
        # self.losses5_kd.update(loss5by4, out_pred[-1].shape[0])

        #total_loss = loss + loss_0

        total_loss = (1 - self.args.alpha) * (loss + middle1_loss  + middle2_loss + middle3_loss )+ \
                        self.args.alpha * (loss1by4 + loss2by4 + loss3by4 )
        # total_loss = loss

        return total_loss
