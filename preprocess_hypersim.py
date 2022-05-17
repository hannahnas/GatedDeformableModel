import numpy as np
import torch
from torchvision.io import read_image
from torchvision.utils import make_grid
import h5py
import matplotlib.pyplot as plt
from os import path
from matplotlib import cm
from utils import *

DATA_FOLDER = '../data/hypersim'
METADATA_FILE = '../metadata_images_split_scene_first20.txt'
RESIZE = 128


def read_metadata(split, metadata_file=METADATA_FILE):
    """
            Obtain metadata file with volume, camera and frame ids for a specific data split.
    """

    array = np.loadtxt(metadata_file, delimiter=',', dtype=str)

    images_metadata = array[1:, :3]
    split_array = array[1:, 5]

    data = images_metadata[split_array == split]

    return data


def filter_and_save_subset(metadata, split, N=576, max_depth=15):

    meta_data = metadata.copy()

    np.random.shuffle(meta_data)
    filtered = []
    max_depth = 0
    for i, [volume_scene, cam, frame] in enumerate(meta_data):
        frame_fill = str(frame).zfill(4)
        color_path = f'{DATA_FOLDER}/{volume_scene}/images/scene_{cam}_final_preview/frame.{frame_fill}.tonemap.jpg'
        depth_path = f'{DATA_FOLDER}/{volume_scene}/images/scene_{cam}_geometry_hdf5/frame.{frame_fill}.depth_meters.hdf5'
        semantic_path = f'{DATA_FOLDER}/{volume_scene}/images/scene_{cam}_geometry_preview/frame.{frame_fill}.semantic.png'

        if path.exists(color_path) and path.exists(depth_path) and path.exists(semantic_path):
            distance = np.array(h5py.File(depth_path, "r")['dataset'])

            if np.isnan(distance).sum() < 10:
                depth = distance2depth(distance)
                if np.all(depth > 0) and np.all(depth < max_depth) and not np.all(depth == 0):
                    rgb = plt.imread(color_path)
                    if not np.all(rgb == 0):
                        filtered.append([volume_scene, cam, frame])
            # print(i)
        if len(filtered) == N:
            print('enough samples found')
            break

    filtered = np.array(filtered)
    np.savetxt(
        f'filtered_metadata_{split}_size{str(N)}.txt', filtered, fmt='%s')

    return filtered


def visual_of_dataset(split, N, rows):
    metadata_file = f'filtered_metadata_{split}_size{str(N)}.txt'
    array = np.loadtxt(metadata_file, dtype=str)
    N = len(array)

    images = []
    depth_maps = []
    for [volume_scene, cam, frame] in array:
        frame_fill = str(frame).zfill(4)
        color_path = f'{DATA_FOLDER}/{volume_scene}/images/scene_{cam}_final_preview/frame.{frame_fill}.tonemap.jpg'

        rgb = read_image(color_path).to(torch.float).unsqueeze(0)
        rgb = resize_and_crop(rgb, interpolation='bilinear').squeeze(0)
        images.append(rgb / 255)

        depth_path = f'{DATA_FOLDER}/{volume_scene}/images/scene_{cam}_geometry_hdf5/frame.{frame_fill}.depth_meters.hdf5'
        distance = np.array(h5py.File(depth_path, "r")['dataset'])
        # Fix NaN values
        if np.isnan(distance).sum() > 0:
            col_mean = np.nanmean(distance, axis=0)
            idx = np.where(np.isnan(distance))
            distance[idx] = np.take(col_mean, idx[1])
        depth = distance2depth(distance)

        depth = torch.Tensor(depth).unsqueeze(0).unsqueeze(1)
        depth = resize_and_crop(depth, interpolation='bilinear')
        depth = depth.squeeze(0)
        depth_maps.append(depth / 15)

    images = torch.Tensor(np.stack(images, axis=0))
    depth_maps = np.stack(depth_maps, axis=0)
    im_grid = make_grid(images, nrow=rows)
    plt.imshow(im_grid.permute(1, 2, 0))
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f'rgb_{split}_set_{str(N)}.png', dpi=400)

    depth_maps = np.apply_along_axis(
        cm.viridis, 0, depth_maps)
    depth_maps = depth_maps[:, :3, 0, :, :]

    depth_grid = make_grid(torch.Tensor(depth_maps),
                           nrow=rows, normalize=True)
    plt.imshow(depth_grid.permute(1, 2, 0))
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f'depth_{split}_set_{str(N)}.png', dpi=400)


if __name__ == '__main__':
    # metadata_train = read_metadata(split='train')
    # filter_and_save_subset(metadata_train, split='train', N=1024)
    # visual_of_dataset('train', 1024, rows=32)

    metadata_val = read_metadata(split='val')
    filter_and_save_subset(metadata_val, split='val', N=128)
    visual_of_dataset('val', 128, rows=8)

    metadata_test = read_metadata(split='test')
    filter_and_save_subset(metadata_val, split='test', N=32)
    visual_of_dataset('test', 32, rows=8)
