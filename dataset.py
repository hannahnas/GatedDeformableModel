# Dataset
# Author: Hannah Min

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import h5py
import matplotlib.pyplot as plt
import cv2
from os import path


class HypersimDataset(Dataset):
    def __init__(
        self,
        data_folder='../data/hypersim',
        metadata_file='../metadata_images_split_scene_first20.txt',
        resize=256,
        split='train'
    ):

        self.resize = resize
        self.data_folder = data_folder
        self.split = split
        if split == 'train':
            N = 512
        elif split == 'val':
            N = 128
        else:
            N = 24
        metadata = self._read_metadata(metadata_file, split)
        self.metadata = self.files_that_exist(metadata, N)

        self.num_images = len(self.metadata)

        self.resize_nearest = transforms.Resize(
            self.resize, interpolation=InterpolationMode.NEAREST)
        self.resize_bilinear = transforms.Resize(
            self.resize, interpolation=InterpolationMode.BILINEAR)
        self.centercrop = transforms.CenterCrop(self.resize)

    def _read_metadata(self, metadata_file, split):
        """
                Obtain metadata file with volume, camera and frame ids for a specific data split.
        """
        array = np.loadtxt(metadata_file, delimiter=',', dtype=str)

        images_metadata = array[1:, :3]
        split_array = array[1:, 5]

        data = images_metadata[split_array == split]
        # np.random.shuffle(data)

        return data

    def files_that_exist(self, metadata, N):
        np.random.shuffle(metadata)
        existing = []
        for i, [volume_scene, cam, frame] in enumerate(metadata):
            frame_fill = str(frame).zfill(4)
            color_path = f'{self.data_folder}/{volume_scene}/images/scene_{cam}_final_preview/frame.{frame_fill}.tonemap.jpg'
            depth_path = f'{self.data_folder}/{volume_scene}/images/scene_{cam}_geometry_hdf5/frame.{frame_fill}.depth_meters.hdf5'
            semantic_path = f'{self.data_folder}/{volume_scene}/images/scene_{cam}_geometry_preview/frame.{frame_fill}.semantic.png'

            if path.exists(color_path) and path.exists(depth_path) and path.exists(semantic_path):
                distance = np.array(h5py.File(depth_path, "r")['dataset'])
                if np.isnan(distance).sum() < 10:
                    existing.append([volume_scene, cam, frame])
                # print(i)
            if len(existing) == N:
                break

        existing = np.array(existing)

        return existing

    def __getitem__(self, index):
        """
                Retrieve color depth and semantic from folder and resize.
        """

        [volume_scene, cam, frame] = self.metadata[index]
        frame_fill = str(frame).zfill(4)

        color_path = f'{self.data_folder}/{volume_scene}/images/scene_{cam}_final_preview/frame.{frame_fill}.tonemap.jpg'
        depth_path = f'{self.data_folder}/{volume_scene}/images/scene_{cam}_geometry_hdf5/frame.{frame_fill}.depth_meters.hdf5'
        semantic_path = f'{self.data_folder}/{volume_scene}/images/scene_{cam}_geometry_preview/frame.{frame_fill}.semantic.png'

        # Read and resize data
        rgb = read_image(color_path).to(torch.float).unsqueeze(0)
        rgb = self.resize_bilinear(rgb)
        rgb = self.centercrop(rgb)
        rgb = rgb.squeeze(0) / 255  # normalize

        # distance = torch.Tensor(h5py.File(depth_path, "r")[
        #                         'dataset']).unsqueeze(0).unsqueeze(1)
        # depth = self.distance_to_depth(distance)
        distance = np.array(h5py.File(depth_path, "r")['dataset'])
        # Fix NaN values
        if np.isnan(distance).sum() > 0:
            col_mean = np.nanmean(distance, axis=0)
            idx = np.where(np.isnan(distance))
            distance[idx] = np.take(col_mean, idx[1])
        #     distance[np.isnan(distance)] = np.nanmean(distance)
        depth = self.distance_to_depth_numpy(distance)

        depth = torch.Tensor(depth).unsqueeze(0).unsqueeze(1)

        depth = self.resize_bilinear(depth)
        depth = self.centercrop(depth)
        # Rescale depth with max depth?
        depth = depth.squeeze(0)  # / 182

        semantic_orig = np.array(Image.open(semantic_path))[
            :, :, :3]  # remove alpha channel
        semantic = torch.Tensor(semantic_orig).permute(2, 0, 1)
        semantic = self.resize_nearest(semantic)
        semantic = self.centercrop(semantic)

        door_mask = cv2.inRange(semantic_orig, (214, 39, 40), (214, 39, 40))
        window_mask = cv2.inRange(
            semantic_orig, (197, 176, 213), (197, 176, 213))
        door_window_mask = torch.Tensor(door_mask + window_mask)
        door_window_mask = door_window_mask.unsqueeze(0).unsqueeze(1)
        door_window_mask = self.resize_nearest(door_window_mask)
        door_window_mask = self.centercrop(door_window_mask).squeeze(0)

        # rgbd_img = torch.cat([rgb, depth], dim=1).squeeze(0)

        img_object = {
            'rgb': rgb,
            'depth': depth,
            'mask': self.generate_mask(self.resize),
            'semantic': semantic,
            'door/window': door_window_mask
        }

        return img_object

    def __len__(self):
        """
                Return the size of the dataset.
        """
        return self.num_images

    def generate_mask(self, img_size):
        """
                Create mask with box in random location.
        """
        H, W = img_size, img_size
        mask = torch.zeros((H, W))
        box_size = round(H * 0.3)

        x_loc = np.random.randint(0, W - box_size)
        y_loc = np.random.randint(0, H - box_size)

        mask[y_loc:y_loc+box_size, x_loc:x_loc+box_size] = 1

        return mask.unsqueeze(0)

    # def distance_to_depth(self, raw_depth):
    #     """
    #             Transform distance from camera to planar depth.
    #     """
    #     width = 1024
    #     height = 768
    #     fltFocal = 886.81
    #     distance = raw_depth

    #     ImageplaneX = torch.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width).reshape(
    #         1, width).repeat(height, 1).type(torch.float32)[:, :, None]
    #     ImageplaneY = torch.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height).reshape(
    #         height, 1).repeat(1, width).type(torch.float32)[:, :, None]
    #     ImageplaneZ = torch.ones([height, width, 1]).fill_(
    #         fltFocal).type(torch.float32)

    #     Imageplane = torch.cat([ImageplaneX, ImageplaneY, ImageplaneZ], 2)

    #     npyDepth = distance / \
    #         torch.linalg.norm(Imageplane, ord=2, dim=2) * fltFocal

    #     return npyDepth

    def distance_to_depth_numpy(self, raw_depth):
        intWidth = 1024
        intHeight = 768
        fltFocal = 886.81

        npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
            1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
        npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(
            intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
        npyImageplaneZ = np.full(
            [intHeight, intWidth, 1], fltFocal, np.float32)
        npyImageplane = np.concatenate(
            [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

        npyDepth = raw_depth / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
        return npyDepth


if __name__ == '__main__':
    split = 'train'
    metadata_file = '../metadata_images_split_scene_first20.txt'
    data_folder = '../data/hypersim'
    resize = 128

    dataset = HypersimDataset(data_folder, metadata_file, resize, split)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for i, batch in enumerate(loader):

        # print('color range', batch['rgbd'][:, :3, :,
        #       :].min(), batch['rgbd'][:, :3, :, :].max())
        # print('amount nan', torch.isnan(batch['rgbd'][:, 3, :, :]).sum())
        check = batch['depth'] * (1 - batch['door/window'])
        print('depth range', check.min(), check.max())
        # # plt.imshow(np.asarray(
        #     batch['rgb(d)'][0, :3, :, :].permute(1, 2, 0), dtype='float'))
        # plt.show()
        # plt.imshow(np.asarray(
        #     batch['rgb(d)'][0, 3:, :, :].permute(1, 2, 0), dtype='float'))
        # plt.show()

        # Komt nan voor of na distance2depth?????/
        # Haalt de door/window mask de nan waardes eruit?
