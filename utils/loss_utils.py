import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=0.000001):        
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


if __name__ == '__main__':
    import torch
    pred = torch.randn(1, 1, 73, 112, 112)
    target = torch.randn(1, 1, 73, 112, 112)
    loss_fn = DiceLoss()

    print(loss_fn(pred, target))