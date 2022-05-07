# The Network

import torch
from torch import nn, optim
import torchvision
import pytorch_lightning as pl
from modules import GatedConv2dWithActivation, DeformableConvWithActivation, UpConvWithActivation, ConvWithActivation, GatedDeformableConvWithActivation
from deform_conv.modules.deform_conv import DeformConvPack
from criterions import mse_loss, ssim_loss, rmse_score, mae_score


class InpaintModel(pl.LightningModule):

    def __init__(self, hyper_params):
        super().__init__()

        self.rgb_encoder = GatedDeformEncoder(hyper_params, 3)
        self.depth_encoder = GatedDeformEncoder(hyper_params, 1)

        self.rgbd_decoder = Decoder(192, 4)

        # self.rgb_decoder = Decoder(768, 3)
        # self.depth_decoder = Decoder(768, 1)

    def forward(self, batch):
        rgb = batch['rgb']
        depth = batch['depth']
        masks = batch['mask']

        color_feat = self.rgb_encoder(rgb, masks)
        depth_feat = self.depth_encoder(depth, masks)

        features = torch.cat((color_feat, depth_feat), dim=1)

        rgbd = self.rgbd_decoder(features)

        # rgb_hat = self.rgb_decoder(features)
        # depth_hat = self.depth_decoder(features)

        return rgbd

    def _get_reconstruction_loss(self, batch):
        gt = torch.cat([batch['rgb'], batch['depth']], dim=1)
        pred = self.forward(batch)
        loss = mse_loss(pred, gt)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)
        with torch.no_grad():
            rgb_gt, depth_gt = batch['rgb'], batch['depth']
            gt = torch.cat([rgb_gt, depth_gt], dim=1)

            pred = self.forward(batch)
            rgb_pred, depth_pred = pred[:, :3, :, :], pred[:, 3:, :, :]

            rmse = rmse_score(depth_pred, depth_gt)
            self.log('RMSE depth', rmse)
            mae = mae_score(depth_pred, depth_gt)
            self.log('MAE depth', mae)
            ssim = ssim_loss(rgb_pred, rgb_gt)
            self.log('SSIM loss', ssim)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)


class GatedDeformEncoder(nn.Module):
    def __init__(self, hyper_params, n_channels):
        super().__init__()
        act_fn = hyper_params['activation']
        size = hyper_params['resize']  # padding based on size !!
        if n_channels == 3:
            c_hid = 32
        if n_channels == 1:
            c_hid = 16

        self.gated_convs = nn.Sequential(
            GatedConv2dWithActivation(
                n_channels+2, c_hid, kernel_size=5, stride=1, padding=1),
            GatedConv2dWithActivation(
                c_hid, c_hid, kernel_size=3, stride=1, padding=1),
            GatedConv2dWithActivation(
                c_hid, 2*c_hid, kernel_size=3, stride=2, padding=1),
            GatedConv2dWithActivation(
                2*c_hid, 2*c_hid, kernel_size=3, stride=1, padding=1),
            GatedConv2dWithActivation(
                2*c_hid, 4*c_hid, kernel_size=3, stride=2, padding=1),
            GatedConv2dWithActivation(
                4*c_hid, 4*c_hid, kernel_size=3, stride=1, padding=1),


        )
        self.deformable_convs = nn.Sequential(
            DeformableConvWithActivation(4*c_hid, 4*c_hid, 3),
            DeformableConvWithActivation(4*c_hid, 4*c_hid, 3),
            DeformableConvWithActivation(4*c_hid, 4*c_hid, 3),
            DeformableConvWithActivation(4*c_hid, 4*c_hid, 3),
            DeformableConvWithActivation(4*c_hid, 4*c_hid, 3),
        )

    # initialize weigts

    def forward(self, imgs, masks):
        masked_imgs = imgs * (1 - masks) + masks
        input_imgs = torch.cat(
            [masked_imgs, masks, torch.full_like(masks, 1.)], dim=1)

        x = self.gated_convs(input_imgs)
        x = self.deformable_convs(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_features=768, out_channels=4):
        super().__init__()

        self.convs = nn.Sequential(
            UpConvWithActivation(in_features, in_features//2, 3, 2),
            ConvWithActivation(in_features//2, in_features//2, 3),
            UpConvWithActivation(in_features//2, in_features//4, 3, 2),
            ConvWithActivation(in_features//4, in_features//4, 3),
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_features//4, out_channels, 3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.convs(x)

        x_hat = self.last_layer(x)
        return x_hat


class Discriminator(nn.Module):
    pass


class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        block = [torchvision.models.vgg16(
            pretrained=True).features[:15].eval()]
        for p in block[0]:
            p.requires_grad = False
        self.block = torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.FloatTensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x-self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(
            224, 224), align_corners=False)
        for block in self.block:
            x = block(x)
        return x


if __name__ == '__main__':
    img = torch.rand(1, 3, 128, 128).to('cuda')
    depth = torch.rand(1, 1, 128, 128).to('cuda')
    mask = torch.randint(0, 2, (1, 1, 128, 128)).to('cuda')
    batch = {
        'rgb': img,
        'depth': depth,
        'mask': mask
    }

    hyper_params = {
        'model name': 'GatedDeformGenerator',
        'epochs': 100,
        'activation': nn.LeakyReLU,
        'resize': 128,
        'batch size': 1
    }

    # model = GatedDeformEncoder(hyper_params, 3).to('cuda')
    # out = model(img, mask)
    # print(out.shape)

    # decoder = Decoder(out_channels=3).to('cuda')
    # out = decoder(out)
    # print(out.shape)

    inpaintmodel = InpaintModel(hyper_params).to('cuda')
    out = inpaintmodel(batch)
    print(out.shape)
