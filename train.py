# train
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
# from network import InpaintModel
from late_fusion_network import LateFusionInpaintModel
from dataset import HypersimDataset
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
CHECKPOINT_PATH = './checkpoints'


class VisulizationCallback(pl.Callback):
    def __init__(self, input, every_n_epochs=1):
        super().__init__()
        self.input = input
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Do image completion
            with torch.no_grad():
                pl_module.eval()
                reconst_rgb, reconst_depth = pl_module(self.input)
                pl_module.train()

            # RGB reconstructions
            rgb = self.input['rgb'] * (1 - self.input['mask'])
            # reconst_rgb = reconst[:, :3, :, :]
            # print('rgb input range', rgb.min(), rgb.max())
            # print('pred range rgb', reconst_rgb.min(), reconst_rgb.max())

            # rgb_show = torch.stack([rgb, reconst_rgb], dim=1).flatten(0, 1)
            # rgb_grid = torchvision.utils.make_grid(
            #     rgb_show, nrow=2, normalize=True, range=(0, 1))
            # trainer.logger.experiment.add_image(
            #     'rgb reconstructions', rgb_grid, global_step=trainer.global_step)

            # DEPTH reconstructions
            depth = self.input['depth']
            # reconst_depth = reconst[:, 3:, :, :]

            # apply cmap for visualization purposes
            depth_map = np.apply_along_axis(cm.viridis, 0, depth.cpu().numpy())

            depth_map = torch.from_numpy(np.squeeze(
                depth_map)).to('cuda') * (1 - self.input['mask'])  # apply mask
            depth_map = depth_map[:, :3, :, :]
            reconst_depth_map = np.apply_along_axis(
                cm.viridis, 0, reconst_depth.cpu().numpy())
            reconst_depth_map = torch.from_numpy(
                np.squeeze(reconst_depth_map)).to('cuda')
            reconst_depth_map = reconst_depth_map[:, :3, :, :]

            # depth_show = torch.stack(
            #     [depth_map, reconst_depth_map], dim=1).flatten(0, 1)
            # depth_grid = torchvision.utils.make_grid(
            #     depth_show, nrow=2, normalize=True, range=(0, 1))
            # trainer.logger.experiment.add_image(
            #     'depth reconstructions', depth_grid, global_step=trainer.global_step)

            results = torch.stack(
                [rgb, reconst_rgb, depth_map, reconst_depth_map], dim=1).flatten(0, 1)
            results_grid = torchvision.utils.make_grid(
                results, nrow=4, normalize=True, range=(0, 1))
            trainer.logger.experiment.add_image(
                'reconstructions', results_grid, global_step=trainer.global_step)


def get_train_images(train_set, num):
    object = {
        'rgb': torch.stack([train_set[i]['rgb'] for i in range(num)], dim=0).to(device),
        'depth': torch.stack([train_set[i]['depth'] for i in range(num)], dim=0).to(device),
        'mask': torch.stack([train_set[i]['mask'] for i in range(num)], dim=0).to(device)
    }

    return object


def visualize_inpainting(model, input_imgs, device, hyper_params, set):
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
    mask = input_imgs['mask'].cpu()
    rgb = input_imgs['rgb'].cpu() * (1 - mask)  # apply mask
    depth = input_imgs['depth'].cpu()
    # reconst_rgb = reconst[:, :3, :, :]
    # reconst_depth = reconst[:, 3:, :, :]

    depth_map = np.apply_along_axis(cm.viridis, 0, depth.numpy())
    depth_map = torch.from_numpy(np.squeeze(
        depth_map)) * (1 - mask)  # apply mask
    depth_map = depth_map[:, :3, :, :]
    reconst_depth_map = np.apply_along_axis(
        cm.viridis, 0, reconst_depth.cpu().numpy())
    reconst_depth_map = torch.from_numpy(
        np.squeeze(reconst_depth_map))
    reconst_depth_map = reconst_depth_map[:, :3, :, :]

    results = torch.stack(
        [rgb, reconst_rgb, depth_map, reconst_depth_map], dim=1).flatten(0, 1)
    results_grid = torchvision.utils.make_grid(
        results, nrow=4, normalize=True, range=(0, 1))

    # Plotting
    results_grid = results_grid.permute(1, 2, 0)
    plt.figure(figsize=(10, 10))
    plt.title(
        f"{set}set {hyper_params['epochs']}epochs_{hyper_params['activation'].__name__}")
    plt.imshow(results_grid)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(
        f"./checkpoints/{hyper_params['model name']}_{set}set_{hyper_params['epochs']}epochs_{hyper_params['activation'].__name__}_input{hyper_params['resize']}", dpi=150)


def train(hyper_params, train_loader, val_loader, test_loader, train_set):

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"{hyper_params['model name']}_epochs{hyper_params['epochs']}_activation{hyper_params['activation'].__name__}"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=hyper_params['epochs'],
                         log_every_n_steps=16,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    VisulizationCallback(get_train_images(
                                        train_set, 8), every_n_epochs=10),
                                    LearningRateMonitor("epoch")])
    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    model = LateFusionInpaintModel()

    trainer.fit(model, train_loader, val_loader)

    # # Test best model on validation and test set
    val_result = trainer.test(
        model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(
        model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}

    return model, result


def run_experiment(hyper_params):
    # Reproducability
    pl.seed_everything(42)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    train_set = HypersimDataset(resize=hyper_params['resize'], split='train')
    print(len(train_set))
    val_set = HypersimDataset(resize=hyper_params['resize'], split='val')
    print(len(val_set))
    test_set = HypersimDataset(resize=hyper_params['resize'], split='test')
    print(len(test_set))

    train_loader = DataLoader(train_set, batch_size=hyper_params['batch size'], shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=4)

    val_loader = DataLoader(val_set, batch_size=hyper_params['batch size'],
                            shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=8,
                             shuffle=False, drop_last=False, num_workers=4)

    model, result = train(hyper_params, train_loader,
                          val_loader, test_loader, train_set)

    for batch in train_loader:
        visualize_inpainting(model, batch, device, hyper_params, 'train')
        break

    for batch in test_loader:
        visualize_inpainting(model, batch, device, hyper_params, 'test')
        break

    return model, result


if __name__ == '__main__':
    hyper_params = {
        'model name': 'GatedDeformLateFusion',
        'epochs': 10,
        'activation': nn.ReLU,
        'resize': 128,
        'batch size': 16
    }

    model, result = run_experiment(hyper_params)

    print(result)
