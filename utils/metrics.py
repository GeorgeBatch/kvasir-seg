import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# to be used for validation during training

def iou_pytorch_eval(outputs: torch.Tensor, labels: torch.Tensor):

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5

    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()


    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou.mean()

# -----------------------------------------------------------------------------
# to be used for validation for trainied models


def iou_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):
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

    return iou.mean()


def dice_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):
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

    return dice.mean()


def precision_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):
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
    precision = (intersection + SMOOTH) / (tpfp + SMOOTH)     # We smooth our devision to avoid 0/0

    return precision.mean()


def recall_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):
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

    return recall.mean()


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

    return acc.mean()



# -----------------------------------------------------------------------------
# Credit to: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return -IoU


class IoUBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoUBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = - (intersection + smooth)/(union + smooth)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        IoU_BCE = BCE + IoU

        return IoU_BCE
