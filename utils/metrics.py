import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

def reduce_metric(metric, reduction='mean'):
    """
    If "sum" or "mean" Reduces a metric tensor in the 0th dimention (batch_size)
    Otherwise returns the metric tensor as is.
    """
    if reduction == 'mean':
        return metric.mean(0)
    elif reduction == 'sum':
        return metric.sum(0)
    elif reduction == 'none':
        return metric
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

# -----------------------------------------------------------------------------
# to be used for validation during training

def iou_pytorch_eval(outputs: torch.Tensor, labels: torch.Tensor, reduction='mean'):

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5

    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # (BATCH, 1, H, W) -> (BATCH, H, W)
    labels = labels.squeeze(1).byte()

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return reduce_metric(iou, reduction)

# -----------------------------------------------------------------------------
# to be used for validation/testing of trainied models

def iou_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor, reduction='mean'):
    # intersection = tp
    # union = tp + fp + fn
    # iou = tp / (tp + fp + fn) = intersection / union

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return reduce_metric(iou, reduction)


def dice_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor, reduction='mean'):
    # intersection = tp
    # union = tp + fp + fn
    # dice = 2 * tp / (2 * tp + fp + fn) = 2*intersection / (intersection + union)

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    dice = (2*intersection + SMOOTH) / (intersection + union + SMOOTH)  # We smooth our devision to avoid 0/0

    return reduce_metric(dice, reduction)


def precision_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor, reduction='mean'):
    # intersection = tp
    # tpfp = tp + fp
    # precision = tp / (tp + fp) = intersection / tpfp

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    tpfp = (labels).float().sum((1, 2))                    # Will be zero if both are 0
    precision = (intersection + SMOOTH) / (tpfp + SMOOTH)  # We smooth our devision to avoid 0/0

    return reduce_metric(precision, reduction)


def recall_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor, reduction='mean'):
    # intersection = tp
    # tpfn = tp + fn
    # recall = tp / (tp + fn) = intersection / tpfn

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    tpfn = (outputs).float().sum((1, 2))                   # Will be zero if both are 0
    recall = (intersection + SMOOTH) / (tpfn + SMOOTH)     # We smooth our devision to avoid 0/0

    return reduce_metric(recall, reduction)


def fbeta_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor, beta:float, reduction='mean'):
    # intersection = tp
    #
    # tpfp = tp + fp
    # precision = tp / (tp + fp) = intersection / tpfp
    #
    # tpfn = tp + fn
    # recall = tp / (tp + fn) = intersection / tpfn
    #
    # fbeta = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
    # https://www.quora.com/What-is-the-F2-score-in-machine-learning

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0

    tpfn = (outputs).float().sum((1, 2))                   # Will be zero if both are 0
    recall = (intersection + SMOOTH) / (tpfn + SMOOTH)     # We smooth our devision to avoid 0/0

    tpfp = (labels).float().sum((1, 2))                    # Will be zero if both are 0
    precision = (intersection + SMOOTH) / (tpfp + SMOOTH)  # We smooth our devision to avoid 0/0

    f_beta = (1 + beta ** 2) * (precision * recall) / ((beta **2 * precision) + recall)

    return reduce_metric(f_beta, reduction)


def accuracy_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    acc = (outputs == labels).float().mean((1, 2))

    return reduce_metric(acc, reduction='mean')

# if we care about both classes
def binary_both_classes_iou_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):
    # intersection = tp
    # union = tp + fp + fn
    # iou = tp / (tp + fp + fn) = intersection / union

    # BATCH x H x W, need because we process images sequentially in a for-loop
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)
    SMOOTH = 1e-8

    # thresholding since that's how we will make predictions on new imputs (class 0)
    outputs0 = outputs < 0.5
    labels0 = labels < 0.5
    intersection = (outputs0 & labels0).float().sum((1, 2))  # Will be zero if Truth=1 or Prediction=1
    union = (outputs0 | labels0).float().sum((1, 2))         # Will be zero if both are 1
    iou0 = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # thresholding since that's how we will make predictions on new imputs (class 1)
    outputs1 = outputs > 0.5
    labels1 = labels > 0.5
    intersection = (outputs1 & labels1).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs1 | labels1).float().sum((1, 2))         # Will be zero if both are 0
    iou1 = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # average iou of both classes - can add unequal weights if we care more about one class
    weighted_iou = (iou0 + iou1) / 2
    return reduce_metric(weighted_iou, reduction='mean')

# -----------------------------------------------------------------------------
# Credit to: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

class IoULoss(nn.Module):
    """
    Assumes shape (B, C, H, W) where C == 1. If C > 1, then we flatten to (B, C*H*W)
    """

    def __init__(self, reduction='mean'):
        super(IoULoss, self).__init__()
        self.reduction = reduction

    def leave_only_batch_and_flatten(self, inputs, targets):
        inputs = inputs.view(inputs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        return inputs, targets

    def forward(self, inputs, targets, smooth=1):
        # inputs are logits, targets are labels in [0, 1]

        # make inputs probabilities and flatten to (batch_size, 256*256)
        inputs, targets = self.leave_only_batch_and_flatten(inputs, targets)
        inputs_after_sigmoid = torch.sigmoid(inputs)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs_after_sigmoid * targets).sum(1)
        total = (inputs_after_sigmoid + targets).sum(1)
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)
        IoU_loss = - IoU

        return reduce_metric(IoU_loss, self.reduction)
    

class BCEWithLogitsLoss(nn.Module):
    """
    Assumes shape (B, C, H, W) where C == 1. If C > 1, then we flatten to (B, C*H*W)
    """
    def __init__(self, reduction='mean'):
        super(BCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none') # will reduce everything together later

    def leave_only_batch_and_flatten(self, inputs, targets):
        # flatten label and prediction tensors: (batch_size, 1, 256, 256) -> (batch_size, 256*256)
        inputs = inputs.view(inputs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        return inputs, targets

    def forward(self, inputs, targets):
        inputs, targets = self.leave_only_batch_and_flatten(inputs, targets)
        BCE_loss = self.BCEWithLogitsLoss(inputs, targets)
        BCE_loss = BCE_loss.mean(1) # mean over all pixels (IoU does this thing automatically): (BATCH, 1*H*W) -> (BATCH, 1)
        return reduce_metric(BCE_loss, reduction=self.reduction)


class IoUBCELoss(IoULoss):
    """
    Assumes shape (B, C, H, W) where C == 1. If C > 1, then we flatten to (B, C*H*W)
    """
    def __init__(self, reduction='mean'):
        super(IoUBCELoss, self).__init__()
        self.reduction = reduction
        self.IoULoss = IoULoss(reduction='none')  # will reduce everything together later
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')

    def leave_only_batch_and_flatten(self, inputs, targets):
        return super(IoUBCELoss, self).leave_only_batch_and_flatten(inputs, targets)

    def forward(self, inputs, targets, smooth=1):
        # inputs are logits, targets are labels in [0, 1]
        
        IoU_loss = self.IoULoss(inputs, targets, smooth=smooth)
        # https://pytorch.org/docs/stable/nn.functional.html#binary-cross-entropy

        inputs, targets = self.leave_only_batch_and_flatten(inputs, targets)
        BCE_loss = self.BCEWithLogitsLoss(inputs, targets)
        BCE_loss = BCE_loss.mean(1) # mean over all pixels (IoU does this thing automatically): (BATCH, 1*H*W) -> (BATCH, 1)
        
        # non-reduced IoU_BCE = non-reduced BCE + non-reduced IoU
        IoU_BCE_loss = BCE_loss + IoU_loss

        return reduce_metric(IoU_BCE_loss, reduction=self.reduction)

# -----------------------------------------------------------------------------

class mIoULossBinary(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(mIoULossBinary, self).__init__()
        self.weight = weight
        self.IoULoss = IoULoss(reduction='none')

    def forward(self, inputs, targets, smooth=1):
        # (BATCH, 1, H, W)
        # we care about both classes represented as 0 and 1 on one the masks

        if self.weight is not None:
            assert self.weight.shape == (targets.shape[1], )
        # make a copy not to change the default weight in the instance of DiceLossMulticlass
        weight = self.weight.copy()

        ipdb.set_trace()

        # invert what is target and what is not
        # 0 -> (-1) * (0 - 1) = 1
        # 1 -> (-1) * (1 - 1) = 0
        targets_inv = (-1) * (targets - 1)

        # rededuction is done at the stage of IoULoss
        if weight is None:
            mIoU = (self.IoULoss(inputs, targets_inv, smooth) + \
                    self.IoULoss(inputs, targets, smooth)) / 2
        else:
            mIoU = (weight[0] * self.IoULoss(inputs, targets_inv, smooth) + \
                    weight[1] * self.IoULoss(inputs, targets, smooth)) / weight.sum()

        return mIoU


# credits for Multiclass implementation to Kenneth Rithvik (in comments)
# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch/comments
class DiceLossMulticlass(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(DiceLossMulticlass, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, smooth=1):
        # inputs, targets of shapes (BATCH, NUM_CLASSES, H, W)

        # check the size of the weight
        if self.weight is not None:
            assert self.weight.shape == (targets.shape[1], )
        # make a copy not to change the default weight in the instance of DiceLossMulticlass
        weight = self.weight.copy()

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction images, leave BATCH and NUM_CLASSES
        # (BATCH, NUM_CLASSES, H, W) -> (BATCH, NUM_CLASSES, H * W)
        inputs = inputs.view(inputs.shape[0],inputs.shape[1],-1)
        targets = targets.view(targets.shape[0],targets.shape[1],-1)

        ipdb.set_trace()

        # get one number per each 2D image/mask pair
        # .sum(2) : (BATCH, NUM_CLASSES, H * W) -> (BATCH, NUM_CLASSES)
        intersection = (inputs * targets).sum(2)
        dice_coef = (2.*intersection + smooth)/(inputs.sum(2) + targets.sum(2) + smooth)
        dice_loss = 1 - dice_coef

        if weight is not None:
            dice_loss = dice_loss * weight

        return reduce_metric(dice_loss, reduction=self.reduction)

