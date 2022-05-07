# criterions
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeometry.losses import SSIM

# Loss functions


def mae_loss(predictions, target):
    loss = F.l1_loss(predictions, target, reduction='none')
    loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
    return loss


def mse_loss(predictions, target):
    loss = F.mse_loss(predictions, target, reduction='none')
    loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
    return loss


# SSIM window=5
def ssim_loss(rgb, target):
    ssim = SSIM(window_size=5, reduction='mean')
    loss = ssim(rgb, target)

    return loss

# Evalutation


def mae_score(predictions, target):

    with torch.no_grad():
        mae = F.l1_loss(predictions, target, reduction='none')
        mae = mae.sum(dim=[1, 2, 3]).mean(dim=[0])

    return mae


def rmse_score(predictions, target):
    with torch.no_grad():
        rmse = torch.sqrt(F.mse_loss(predictions, target, reduction='none'))
        rmse = rmse.sum(dim=[1, 2, 3]).mean(dim=[0])

    return rmse


def delta_acc(depth, target,  threshold):
    _, _, W, H = depth.shape

    with torch.no_grad():
        abs_error = (depth - target).abs()
        delta_acc = (abs_error < threshold).sum(dim=(2, 3)) / (W*H)
        delta_acc = delta_acc.mean()

    return delta_acc


def perceptual_loss(predictions, target):
    pass

# PSNR

# Dice

# focal loss


if __name__ == '__main__':
    img = torch.rand(1, 3, 128, 128)
    img2 = torch.rand(1, 3, 128, 128)
    depth = torch.rand(4, 1, 128, 128)
    depth2 = torch.rand(4, 1, 128, 128)

    # delta_acc(img, img2, 0.5)
    # print(mae_score(img, img2))
    print(delta_acc(depth, depth2, 0.5))
