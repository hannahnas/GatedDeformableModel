from late_fusion_network import LateFusionInpaintModel
from dataset import HypersimDataset
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import os
import open3d as o3d
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from utils import *
from plotting import *

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DATA_FOLDER = '../data/hypersim'

def inference(model, data, device=device):
    # Reconstruct images
    model.eval()
    model.to(device)
    data['rgb'] = data['rgb'].to(device)
    data['depth'] = data['depth'].to(device)
    data['mask'] = data['mask'].to(device)
    print(data['rgb'].min(), data['rgb'].max())

    with torch.no_grad():
        rgb_pred, depth_pred = model(data)
        # rgb_pred = torch.clamp(rgb_pred, min=0, max=1)
        # depth_pred = torch.clamp(depth_pred, min=0, max=1)
        rgb_out = (data['mask'] * rgb_pred) + ((1 - data['mask']) * data['rgb'])
        depth_out = data['mask'] * depth_pred + (1 - data['mask']) * data['depth']
    rgb_out, depth_out = rgb_out.cpu(), depth_out.cpu()

    # plt.imshow(rgb_pred[0].permute(1, 2, 0).cpu())
    # plt.show()
    return rgb_out, depth_out

if __name__ == '__main__':
    # model_path = './checkpoints/GatedDeformLateFusion_epochs200_activationReLU/lightning_logs/version_0/checkpoints/epoch=128-step=10964.ckpt'
    model_path = './checkpoints/GatedDeformLateFusion_epochs600_activationReLU/lightning_logs/version_0/checkpoints/epoch=535-step=45559.ckpt'
    # model_path = './checkpoints/MaskedGatedDeformLateFusion_epochs300_activationReLU/lightning_logs/version_0/checkpoints/epoch=269-step=34559.ckpt'

    hyper_params = {
        'model name': 'GatedDeformLateFusion',
        'epochs': 128,
        'activation': nn.ReLU,
        'resize': 128,
        'batch size': 8
    }

    if os.path.isfile(model_path):
        print(f"Found pretrained model at {model_path}, loading...")
        model = LateFusionInpaintModel(fusion=True)
        model = model.load_from_checkpoint(model_path) # Automatically loads the model with the saved hyperparameters
        model = model.to(device)
        model.eval()

    # metadata_file='./metadata/filtered_metadata_test_size32.txt'
    # test_set = HypersimDataset(metadata_file=metadata_file, resize=128)
    metadata_file='./metadata/filtered_metadata_train_size1024.txt'
    train_set = HypersimDataset(metadata_file=metadata_file, resize=128)
    train_loader = loader = DataLoader(train_set, batch_size=8, shuffle=True)
    # data = get_images(train_set, 4, device='cpu')
    # rgb_pred, depth_pred = inference(model, data)
    # visualize_results(data, rgb_pred, depth_pred, hyper_params, set='train')


    for batch in train_loader:
        batch['rgb'] = batch['rgb'].to(device)
        batch['depth'] = batch['depth'].to(device)
        batch['mask'] = batch['mask'].to(device)

        reconst_rgb, reconst_depth = model(batch)
        completed_rgb = batch['rgb'] * (1 - batch['mask']) + reconst_rgb * batch['mask'] 
        completed_depth = batch['depth'] * (1 - batch['mask']) + reconst_depth * batch['mask']

        completed_rgb, completed_depth = completed_rgb.permute(0, 2, 3, 1).cpu().detach().numpy()[0], completed_depth.cpu().detach().numpy()[0, 0]

        # rgb_gt, depth_gt = batch['rgb'].permute(0, 2, 3, 1).cpu().detach().numpy()[0], batch['depth'].cpu().detach().numpy()[0, 0]
        print(completed_rgb.shape)
        print(completed_depth.shape)
        plot_RGBD_o3d(completed_rgb*255, completed_depth)
        break
    #     # plot_RGBD(rgb_gt, depth_gt)
    #     color_gt, depth_gt = read_hypersim_numpy(DATA_FOLDER, volume_scene='ai_007_004', cam='cam_02', frame=10)
    #     matplotlib_3D_plot(rgb_gt, depth_gt)


    #     break
    

    