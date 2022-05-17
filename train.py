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
from utils import *
from plotting import *


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
            rgb_input = self.input['rgb'] * (1 - self.input['mask'])
            predicted_rgb = reconst_rgb * self.input['mask']
            completed_rgb = rgb_input + predicted_rgb

            # DEPTH reconstructions
            depth = self.input['depth']
            # apply cmap for visualization purposes
            depth_map = np.apply_along_axis(cm.viridis, 0, depth.cpu().numpy())
            depth_map_input = torch.from_numpy(np.squeeze(
                depth_map)).to('cuda') * (1 - self.input['mask'])  # apply mask
            depth_map_input = depth_map_input[:, :3, :, :]
            reconst_depth_map = np.apply_along_axis(
                cm.viridis, 0, reconst_depth.cpu().numpy())
            reconst_depth_map = torch.from_numpy(
                np.squeeze(reconst_depth_map)).to('cuda')
            reconst_depth_map = reconst_depth_map[:, :3, :, :]
            predicted_depth = reconst_depth_map * self.input['mask']
            completed_depth = predicted_depth + depth_map_input

            results = torch.stack(
                [rgb_input, completed_rgb, depth_map_input, completed_depth], dim=1).flatten(0, 1)
            results_grid = torchvision.utils.make_grid(
                results, nrow=4, range=(0, 1))
            trainer.logger.experiment.add_image(
                'reconstructions', results_grid, global_step=trainer.global_step)


def train(hyper_params, train_loader, val_loader, test_loader, train_set):

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"{hyper_params['model name']}_epochs{hyper_params['epochs']}_activation{hyper_params['activation'].__name__}"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=hyper_params['epochs'],
                         log_every_n_steps=16,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                                    VisulizationCallback(get_images(
                                        train_set, 8, device), every_n_epochs=10),
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

    train_set = HypersimDataset(
        metadata_file='./metadata/filtered_metadata_train_size1024.txt',
        resize=hyper_params['resize']
    )
    print('N datapoints train set:', len(train_set))

    val_set = HypersimDataset(
        metadata_file='./metadata/filtered_metadata_val_size128.txt',
        resize=hyper_params['resize']
    )
    print('N datapoints validation set:', len(val_set))

    test_set = HypersimDataset(
        metadata_file='./metadata/filtered_metadata_test_size32.txt',
        resize=hyper_params['resize']
    )
    print('N datapoints test set:', len(test_set))

    train_loader = DataLoader(train_set, batch_size=hyper_params['batch size'], shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=4)

    val_loader = DataLoader(val_set, batch_size=hyper_params['batch size'],
                            shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=8,
                             shuffle=False, drop_last=False, num_workers=4)

    model, result = train(hyper_params, train_loader,
                          val_loader, test_loader, train_set)

    for batch in train_loader:
        save_inpainting_results(model, batch, device, hyper_params, 'train')
        break

    for batch in test_loader:
        save_inpainting_results(model, batch, device, hyper_params, 'test')
        break

    return model, result


if __name__ == '__main__':
    hyper_params = {
        'model name': 'TestVisGatedDeformLateFusion',
        'epochs': 50,
        'activation': nn.ReLU,
        'resize': 128,
        'batch size': 8
    }

    model, result = run_experiment(hyper_params)

    print(result)
