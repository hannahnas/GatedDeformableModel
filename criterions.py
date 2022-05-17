# criterions
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeometry.losses import SSIM
import numpy as np

# Loss functions


def mae_loss(predictions, targets, loss_masks, door_window=None):
    # ignore door/window pixels
    if torch.is_tensor(door_window):
        predictions = predictions * (1 - door_window)
        targets = targets * (1 - door_window)

    # Only consider to be inpainted region
    predictions = select_region(predictions, loss_masks)
    targets = select_region(targets, loss_masks)
    loss = F.l1_loss(predictions, targets, reduction='mean')
    # loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
    return loss


def mse_loss(predictions, target):
    loss = F.mse_loss(predictions, target, reduction='none')
    loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
    return loss


# SSIM window=5
def ssim_loss(rgb, target, loss_masks):
    # Only consider to be inpainted region
    rgb = select_region(rgb, loss_masks)
    target = select_region(target, loss_masks)
    ssim = SSIM(window_size=5, reduction='mean', max_val=1)
    loss = ssim(rgb, target)
    # loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])

    return loss


def select_region(inputs, loss_masks):
    B, C, _, _ = inputs.shape
    if C == 3:
        loss_masks = loss_masks.repeat(1, 3, 1, 1)
    cropped = inputs[loss_masks.bool()].reshape(B, C, 48, 48)

    return cropped

# def bbox2(mask):
#     rows = torch.any(mask, axis=3)
#     cols = torch.any(mask, axis=2)
#     ymin, ymax = np.where(rows)[[0, -1]]
#     xmin, xmax = np.where(cols)[[0, -1]]
#     return xmin, xmax, ymin, ymax


# def generate_mask(img_size):
#     """
#             Create mask with box in random location.
#     """
#     H, W = img_size, img_size
#     mask = torch.zeros((H, W))
#     box_size = round(H * 0.3)

#     x_loc = np.random.randint(0, W - box_size)
#     y_loc = np.random.randint(0, H - box_size)
#     print('x', x_loc, x_loc+box_size)
#     print('y', y_loc, y_loc+box_size)

#     mask[y_loc:y_loc+box_size, x_loc:x_loc+box_size] = 1

#     return mask.unsqueeze(0)

# Evaluation


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


# def perceptual_loss(predictions, target):
#     pass


if __name__ == '__main__':
    img = torch.rand(1, 3, 128, 128)
    img2 = torch.rand(1, 3, 128, 128)
    depth = torch.rand(4, 1, 128, 128)
    depth2 = torch.rand(4, 1, 128, 128)

    # l1 = mae_loss(img, img2)
    # print(l1)
    # ssim = ssim_loss(img, img2)
    # print(ssim)
    # mask = generate_mask(128)
    # mask2 = generate_mask(128)
    # masks = torch.cat([mask, mask2], dim=0)
    # xmin, xmax, ymin, ymax = bbox2(masks)
    # print('x', xmin, xmax)
    # print('y', ymin, ymax)
    # delta_acc(img, img2, 0.5)
    # print(mae_score(img, img2))
    # print(delta_acc(depth, depth2, 0.5))
