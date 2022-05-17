import numpy as np
from matplotlib import cm
import torch
import matplotlib.pyplot as plt
import torchvision
import open3d as o3d
from utils import *


def save_inpainting_results(model, input_imgs, device, hyper_params, set):
    # Reconstruct images
    model.eval()
    model.to(device)
    input_imgs['rgb'] = input_imgs['rgb'].to(device)[:4]
    input_imgs['depth'] = input_imgs['depth'].to(device)[:4]
    input_imgs['mask'] = input_imgs['mask'].to(device)[:4]
    with torch.no_grad():
        reconst_rgb, reconst_depth = model(input_imgs)
    reconst_rgb, reconst_depth = reconst_rgb.cpu(), reconst_depth.cpu()

   # RGB reconstructions
    rgb = input_imgs['rgb'].cpu()
    mask = input_imgs['mask'].cpu()
    rgb_input = rgb * (1 - mask)
    predicted_rgb = reconst_rgb * mask
    completed_rgb = rgb_input + predicted_rgb

    # DEPTH reconstructions
    depth = input_imgs['depth'].cpu()
    # apply cmap for visualization purposes
    depth_map = np.apply_along_axis(cm.viridis, 0, depth.numpy())
    depth_map = torch.from_numpy(np.squeeze(
        depth_map))[:, :3, :, :]
    depth_map_input = depth_map * (1 - mask)  # apply mask
    depth_map_input = depth_map_input
    reconst_depth_map = np.apply_along_axis(
        cm.viridis, 0, reconst_depth.numpy())
    reconst_depth_map = torch.from_numpy(
        np.squeeze(reconst_depth_map))
    reconst_depth_map = reconst_depth_map[:, :3, :, :]
    predicted_depth = reconst_depth_map * mask
    completed_depth = predicted_depth + depth_map_input

    results = torch.stack(
        [rgb_input, completed_rgb, rgb, depth_map_input, completed_depth, depth_map], dim=1).flatten(0, 1)

    results_grid = torchvision.utils.make_grid(
        results, nrow=6, range=(0, 1))

    # Plotting
    results_grid = results_grid.permute(1, 2, 0)
    plt.figure(figsize=(10, 10))
    plt.title(
        '|  input RGB  |  pred RGB  |  gt RGB  |  depth input  |  depth pred  |  gt depth  |')
    plt.imshow(results_grid)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(
        f"./checkpoints/{hyper_params['model name']}_{set}set_{hyper_params['epochs']}epochs_{hyper_params['activation'].__name__}_input{hyper_params['resize']}", dpi=150)


def visualize_results(batch, rgb_pred, depth_pred, hyper_params, set):
    rgb_gt = batch['rgb'].cpu()
    depth_gt = batch['depth'].cpu()
    mask = batch['mask'].cpu()

    masked_rgb = rgb_gt * (1 - mask)
    print(rgb_pred.min())

    # apply color map
    depth_gt = np.apply_along_axis(cm.viridis, 0, depth_gt.numpy())
    depth_gt = torch.from_numpy(np.squeeze(depth_gt))[
        :, :3, :, :]  # remove alpha channel
    masked_depth = depth_gt * (1 - mask)  # apply mask

    depth_pred = np.apply_along_axis(cm.viridis, 0, depth_pred.numpy())
    depth_pred = torch.from_numpy(
        np.squeeze(depth_pred))[:, :3, :, :]  # remove alpha channel

    results = torch.stack(
        [masked_rgb, rgb_pred, rgb_gt, masked_depth, depth_pred, depth_gt], dim=1).flatten(0, 1)
    results_grid = torchvision.utils.make_grid(
        results, nrow=6, normalize=True, value_range=(0, 1))

    results_grid = results_grid.permute(1, 2, 0)
    plt.figure(figsize=(10, 10))
    plt.suptitle(
        f"{set}set {hyper_params['epochs']}epochs_{hyper_params['activation'].__name__}")
    plt.title(
        '|  input RGB  |  pred RGB  |  gt RGB  |  depth input  |  depth pred  |  gt depth  |')
    plt.imshow(results_grid)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # plt.savefig(
    #     f"./checkpoints/{hyper_params['model name']}_{set}set_{hyper_params['epochs']}epochs_{hyper_params['activation'].__name__}_input{hyper_params['resize']}", dpi=150)


def plot_RGBD(color_raw, depth_raw):
    cam = o3d.camera.PinholeCameraIntrinsic()

    cam.set_intrinsics(
        width=1024,
        height=768,
        fx=886.81,
        fy=886.81,
        cx=511.5,
        cy=383.5
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=True)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic=cam)

    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #     rgbd_image, intrinsic=o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                   [0, 0, -1, 0], [0, 0, 0, 1]])
    print(pcd)
    o3d.visualization.draw_geometries(
        [pcd])


def plot_RGBD_o3d(color, depth):

    color = o3d.geometry.Image(np.ascontiguousarray(color).astype(np.uint8))
    depth = o3d.geometry.Image(np.ascontiguousarray(depth))
    cam = o3d.camera.PinholeCameraIntrinsic()

    cam.set_intrinsics(
        width=128,
        height=128,
        fx=147.22,
        fy=147.80,
        cx=64.5,
        cy=64.5
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic=cam)

    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #     rgbd_image, intrinsic=o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                   [0, 0, -1, 0], [0, 0, 0, 1]])
    print(pcd)
    o3d.visualization.draw_geometries(
        [pcd])


def matplotlib_3D_plot(img, depth):
    img = np.flipud(img * 255)
    depth = np.flipud(depth)

    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection='3d')
    STEP = 5
    for x in range(0, img.shape[0], STEP):
        for y in range(0, img.shape[1], STEP):
            ax.scatter(
                depth[x, y], y, x,
                c=[tuple(img[x, y, :3]/255)], s=3)
    ax.view_init(5, 100)
    plt.show()


if __name__ == '__main__':
    color, depth = read_hypersim_numpy(
        '../data/hypersim', volume_scene='ai_007_004', cam='cam_02', frame=10)
    color = torch.Tensor(color).permute(2, 0, 1).unsqueeze(0)
    color = resize_and_crop(color, 128, 'bilinear')
    color = color.squeeze(0).permute(1, 2, 0)

    depth = torch.Tensor(depth).unsqueeze(0).unsqueeze(0)
    depth = resize_and_crop(torch.Tensor(depth), 128, 'bilinear')
    depth = depth.squeeze(0).squeeze(0)
    print(color.shape)
    print(depth.shape)

    plot_RGBD_o3d(color, depth)
