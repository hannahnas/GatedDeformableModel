import torch
from torch import nn, optim
import torchvision
import pytorch_lightning as pl
from modules import GatedConv2dWithActivation, DeformableConvWithActivation, UpConvWithActivation, ConvWithActivation, GatedDeformConvWithActivation
from deform_conv.modules.deform_conv import DeformConvPack
from criterions import mse_loss, ssim_loss, mae_loss


class LateFusionInpaintModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.rgb_enc_dec = GatedDeformEncDec(
            in_channels=3, latent=32, upsample_mode='bilinear')
        self.depth_enc_dec = GatedDeformEncDec(
            in_channels=1, latent=16, upsample_mode='nearest')

        self.inpaint_rgb = Inpainter(in_channels=48, out_channels=3)

        self.inpaint_depth = Inpainter(in_channels=48, out_channels=1)

    def forward(self, batch):
        rgb = batch['rgb']
        depth = batch['depth']
        masks = batch['mask']

        color_feat = self.rgb_enc_dec(rgb, masks)
        depth_feat = self.depth_enc_dec(depth, masks)

        features = torch.cat((color_feat, depth_feat), dim=1)

        rgb_hat = self.inpaint_rgb(features)
        depth_hat = self.inpaint_depth(features)

        # rgbd = torch.cat([rgb_hat, depth_hat], dim=1)

        return rgb_hat, depth_hat

    def _get_reconstruction_loss(self, batch):
        rgb_gt, depth_gt = batch['rgb'], batch['depth']
        rgb_pred, depth_pred = self.forward(batch)
        l1_rgb = mae_loss(rgb_pred, rgb_gt)
        self.log('L1 RGB', l1_rgb)
        ssim_rgb = ssim_loss(rgb_pred, rgb_gt)
        self.log('SSIM RGB', ssim_rgb)
        l1_depth = mae_loss(depth_pred, depth_gt)
        self.log('L1 depth', l1_depth)

        loss = 0.5 * l1_rgb + 0.5 * ssim_rgb + l1_depth

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs

        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                  mode='min',
        #                                                  factor=0.2,
        #                                                  patience=20,
        #                                                  min_lr=5e-5)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return {"optimizer": optimizer, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)


class GatedDeformEncDec(nn.Module):
    def __init__(self, in_channels, latent, upsample_mode):
        super().__init__()

        # Encoder
        self.EncBlock0 = nn.Sequential(
            GatedDeformConvWithActivation(
                in_channels, latent, kernel_size=5, stride=1, padding=2),
            GatedDeformConvWithActivation(
                latent, latent, kernel_size=3, stride=1, padding=1),
        )
        self.EncBlock1 = nn.Sequential(
            GatedDeformConvWithActivation(
                latent, 2*latent, kernel_size=3, stride=2, padding=1),
            GatedDeformConvWithActivation(
                2*latent, 2*latent, kernel_size=3, stride=1, padding=1)
        )

        self.EncBlock2 = nn.Sequential(
            GatedDeformConvWithActivation(
                2*latent, 4*latent, kernel_size=3, stride=2, padding=1),
            GatedDeformConvWithActivation(
                4*latent, 4*latent, kernel_size=3, stride=1, padding=1)
        )

        # Decoder
        self.DecBlock2 = nn.Sequential(
            UpConvWithActivation(4*latent, 2*latent, 3, 2, upsample_mode),
            ConvWithActivation(2*latent, 2*latent, 3),
        )

        self.DecBlock1 = nn.Sequential(
            UpConvWithActivation(2*2*latent, latent, 3, 2, upsample_mode),
            ConvWithActivation(latent, latent, 3),
        )

        self.DecBlock0 = nn.Sequential(
            ConvWithActivation(2*latent, latent, 3),
            ConvWithActivation(latent, latent, 3)
        )

    def forward(self, inputs, masks):
        masked_imgs = inputs * (1 - masks) + masks
        # input_imgs = torch.cat(
        #     [masked_imgs, masks, torch.full_like(masks, 1.)], dim=1)

        conv0 = self.EncBlock0(masked_imgs)  # B, L, 128, 128

        conv1 = self.EncBlock1(conv0)  # B, 2L, 64, 64

        conv2 = self.EncBlock2(conv1)  # B, 4L, 32, 32

        upconv2 = self.DecBlock2(conv2)  # B, 2L, 64, 64

        skipconv2 = torch.cat([upconv2, conv1], dim=1)  # B, 4L, 64, 64

        upconv1 = self.DecBlock1(skipconv2)  # B, L, 128, 128

        skipconv1 = torch.cat([upconv1, conv0], dim=1)  # B, 2L, 128, 128

        final = self.DecBlock0(skipconv1)  # B, L, 128, 128

        return final


class Inpainter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            ConvWithActivation(in_channels, in_channels//2, 3),
            ConvWithActivation(in_channels//2, in_channels//4, 3),
            nn.Conv2d(in_channels//4, out_channels, 3, padding=1)
        )

    def forward(self, x):

        return self.net(x)


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
        'model name': 'GatedDeformLateFusionModel',
        'epochs': 100,
        'activation': nn.ReLU,
        'resize': 128,
        'batch size': 1
    }

    model = LateFusionInpaintModel().to('cuda')
    rgbd = model(batch)
    print(rgbd.shape)
