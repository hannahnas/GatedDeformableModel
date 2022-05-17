import numpy as np
import torch
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
import h5py
import open3d as o3d


def distance2depth(distance):
    """
    Script to convert distances to the camera center to planar depth.
    https://github.com/apple/ml-hypersim/issues/9#issuecomment-754935697
    Input:
        Array (numpy)
    Returns:
        Array (numpy)
    """
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

    npyDepth = distance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth


def get_images(train_set, num, device):
    print(device)
    object = {
        'rgb': torch.stack([train_set[i]['rgb'] for i in range(num)], dim=0).to(device),
        'depth': torch.stack([train_set[i]['depth'] for i in range(num)], dim=0).to(device),
        'mask': torch.stack([train_set[i]['mask'] for i in range(num)], dim=0).to(device)
    }

    return object


def resize_and_crop(image, resize, interpolation):
    resize_nearest = T.Resize(
        resize, interpolation=T.InterpolationMode.NEAREST)
    resize_bilinear = T.Resize(
        resize, interpolation=T.InterpolationMode.BILINEAR)
    centercrop = T.CenterCrop(resize)

    if interpolation == 'nearest':
        image = resize_nearest(image)
    elif interpolation == 'bilinear':
        image = resize_bilinear(image)
    image = centercrop(image)
    return image


def create_door_window_mask(semantic):
    """
    Create door/window mask from single semantic annotation image.
    """
    door_mask = cv2.inRange(semantic, (214, 39, 40), (214, 39, 40))
    window_mask = cv2.inRange(
        semantic, (197, 176, 213), (197, 176, 213))
    door_window_mask = torch.Tensor(door_mask + window_mask)
    return door_window_mask


def read_hypersim_numpy(data_folder, volume_scene, cam, frame):
    frame_fill = str(frame).zfill(4)
    color_path = f'{data_folder}/{volume_scene}/images/scene_{cam}_final_preview/frame.{frame_fill}.tonemap.jpg'
    depth_path = f'{data_folder}/{volume_scene}/images/scene_{cam}_geometry_hdf5/frame.{frame_fill}.depth_meters.hdf5'
    img = plt.imread(color_path)
    distance = np.array(h5py.File(depth_path, "r")['dataset'])
    depth = distance2depth(distance)
    return img, depth


def read_hypersim_o3dImage(data_folder, volume_scene, cam, frame):
    frame_fill = str(frame).zfill(4)
    color_path = f'{data_folder}/{volume_scene}/images/scene_{cam}_final_preview/frame.{frame_fill}.tonemap.jpg'
    depth_path = f'{data_folder}/{volume_scene}/images/scene_{cam}_geometry_hdf5/frame.{frame_fill}.depth_meters.hdf5'
    color_Img = o3d.io.read_image(color_path)

    distance = np.array(h5py.File(depth_path, "r")['dataset'])
    # Fix NaN values
    if np.isnan(distance).sum() > 0:
        col_mean = np.nanmean(distance, axis=0)
        idx = np.where(np.isnan(distance))
        distance[idx] = np.take(col_mean, idx[1])
    depth_raw = distance2depth(distance)
    depth_Img = o3d.geometry.Image(depth_raw)

    return color_Img, depth_Img
