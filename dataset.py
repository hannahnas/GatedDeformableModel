# Dataset
# Author: Hannah Min

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.io import read_image
import h5py
from utils import *


class HypersimDataset(Dataset):
    def __init__(
        self,
        data_folder='../data/hypersim',
        metadata_file='./metadata/filtered_metadata_train_size1024.txt',
        resize=128
    ):
        self.resize = resize
        self.data_folder = data_folder
        self.metadata = np.loadtxt(metadata_file, dtype=str)
        self.num_images = len(self.metadata)

    def __getitem__(self, index):
        """
                Retrieve color depth and semantic from folder and resize.
        """

        [volume_scene, cam, frame] = self.metadata[index]
        frame_fill = str(frame).zfill(4)

        color_path = f'{self.data_folder}/{volume_scene}/images/scene_{cam}_final_preview/frame.{frame_fill}.tonemap.jpg'
        depth_path = f'{self.data_folder}/{volume_scene}/images/scene_{cam}_geometry_hdf5/frame.{frame_fill}.depth_meters.hdf5'
        semantic_path = f'{self.data_folder}/{volume_scene}/images/scene_{cam}_geometry_preview/frame.{frame_fill}.semantic.png'

        # Color
        rgb = read_image(color_path).to(torch.float).unsqueeze(0)
        rgb = resize_and_crop(rgb, self.resize, 'bilinear')
        # Normalize pixel values
        rgb = rgb.squeeze(0) / 255

        # Depth
        distance = np.array(h5py.File(depth_path, "r")['dataset'])
        # Fill NaN values
        if np.isnan(distance).sum() > 0:
            col_mean = np.nanmean(distance, axis=0)
            idx = np.where(np.isnan(distance))
            distance[idx] = np.take(col_mean, idx[1])
        depth = distance2depth(distance)
        depth = torch.Tensor(depth).unsqueeze(0).unsqueeze(1)
        depth = resize_and_crop(depth, self.resize, 'bilinear')
        # Rescale depth with max depth?
        depth = depth.squeeze(0) / 15

        # Obtain mask for doors and windows
        semantic_orig = np.array(Image.open(semantic_path))[
            :, :, :3]  # remove alpha channel
        semantic = torch.Tensor(semantic_orig).permute(2, 0, 1)
        semantic = resize_and_crop(semantic, self.resize, 'nearest')
        door_window_mask = create_door_window_mask(semantic_orig)
        door_window_mask = door_window_mask.unsqueeze(0).unsqueeze(1)
        door_window_mask = resize_and_crop(
            door_window_mask, self.resize, 'nearest').squeeze(0)

        mask, mask_with_border = self.generate_masks(self.resize)

        img_object = {
            'rgb': rgb,
            'depth': depth,
            'mask': mask,
            'loss mask': mask_with_border,
            # 'semantic': semantic,
            'door/window': door_window_mask
        }

        return img_object

    def __len__(self):
        """
                Return the size of the dataset.
        """
        return self.num_images

    def generate_masks(self, img_size, border=5):
        """
                Create mask with box in random location.
                Create another slightly bigger mask in the same location.
        """
        H, W = img_size, img_size
        mask = torch.zeros((H, W))
        mask_with_border = torch.zeros((H, W))
        box_size = round(H * 0.3)

        x_loc = np.random.randint(border, W - box_size - border)
        y_loc = np.random.randint(border, H - box_size - border)

        mask[y_loc:y_loc+box_size, x_loc:x_loc+box_size] = 1

        mask_with_border[y_loc-border:y_loc+box_size +
                         border, x_loc-border:x_loc+box_size+border] = 1

        return mask.unsqueeze(0), mask_with_border.unsqueeze(0)


if __name__ == '__main__':
    data_folder = '../data/hypersim'
    metadata_file = './metadata/filtered_metadata_train_size1024.txt'
    resize = 128

    dataset = HypersimDataset(data_folder, metadata_file, resize)

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for i, batch in enumerate(loader):
        print(batch['rgb'].shape)
        print(batch['depth'].shape)
        bool_index = batch['loss mask'].repeat(
            1, 3, 1, 1)
        print(bool_index.shape)
        cropped = batch['rgb'][bool_index.bool()].reshape(8, 3, 48, 48)
        print(cropped.shape)
        fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(batch['mask'][0, 0])
        ax[2].imshow(cropped[0].permute(1, 2, 0))
        ax[1].imshow(batch['loss mask'][0, 0])
        ax[0].imshow(batch['rgb'][0].permute(1, 2, 0))
        plt.show()
        break
