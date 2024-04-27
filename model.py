"""
*Preliminary* pytorch implementation.

Networks for voxelmorph model

In general, these are fairly specific architectures that were designed for the presented papers.
However, the VoxelMorph concepts are not tied to a very particular architecture, and we
encourage you to explore architectures that fit your needs.
see e.g. more powerful unet function in https://github.com/adalca/neuron/blob/master/neuron/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
import math

############################################################################# Model
class LKG_Net(nn.Module):
    def __init__(self,
                 vol_shape,
                 parallel_sizes=None,
                 start_channel=None,
                 remain_channels=None,
                ):
        super(LKG_Net, self).__init__()

        self.dim = len(vol_shape)
        self.encoder_1 = nn.Sequential(
            ConvBlock(in_channels=2, out_channels=start_channel),
            LargeGhostBlock(in_channels=start_channel, out_channels=start_channel, parallel_sizes=parallel_sizes)
        )
        self.encoder_2 = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(start_channel, start_channel * 2),
            LargeGhostBlock(in_channels=start_channel * 2, out_channels=start_channel * 2,
                            parallel_sizes=parallel_sizes)

        )
        self.encoder_3 = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(start_channel * 2, start_channel * 2),
            LargeGhostBlock(in_channels=start_channel * 2, out_channels=start_channel * 2,
                            parallel_sizes=parallel_sizes)

        )
        self.encoder_4 = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(start_channel * 2, start_channel * 2),
            LargeGhostBlock(in_channels=start_channel * 2, out_channels=start_channel * 2,
                            parallel_sizes=parallel_sizes)

        )
        self.decoder_1 = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(start_channel * 2, start_channel * 2),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.decoder_2 = nn.Sequential(
            ConvBlock(start_channel * 4, start_channel * 2),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.decoder_3 = nn.Sequential(
            ConvBlock(start_channel * 4, start_channel * 2),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.decoder_4 = nn.Sequential(
            ConvBlock(start_channel * 4, start_channel * 2),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.remains=nn.ModuleList()
        pre_nf=start_channel * 3
        for remain_channel in remain_channels:
            self.remains.append(ConvBlock(pre_nf, remain_channel))
            pre_nf = remain_channel

        self.flow=nn.Conv3d(in_channels=pre_nf,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, source, target):
        x = torch.cat([source, target], dim=1)
        enc_1 = self.encoder_1(x)
        enc_2 = self.encoder_2(enc_1)
        enc_3 = self.encoder_3(enc_2)
        enc_4 = self.encoder_4(enc_3)

        dec_1 = self.decoder_1(enc_4)
        dec_1 = torch.cat([enc_4, dec_1], dim=1)
        dec_2 = self.decoder_2(dec_1)
        dec_2 = torch.cat([enc_3, dec_2], dim=1)
        dec_3 = self.decoder_3(dec_2)
        dec_3 = torch.cat([enc_2, dec_3], dim=1)
        dec_4 = self.decoder_4(dec_3)
        dec_4 = torch.cat([enc_1, dec_4], dim=1)

        y = dec_4
        for conv in self.remains:
            y = conv(y)

        phi = self.flow(y)

        return phi

############################################################################# Block


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, ksize=3, stride=1,padding=1,mode='origin',batchnorm=False):
        super().__init__()
        if mode == 'origin':
            self.main = nn.Conv3d(in_channels, out_channels, ksize, stride, padding)
        elif mode == 'dw':
            self.main = nn.Conv3d(in_channels, out_channels, ksize, stride, padding,groups=in_channels,bias=False)
        #self.batchnorm = nn.BatchNorm3d(out_channels)
        #self.activation = nn.LeakyReLU(0.2)
        if batchnorm == True:
            self.remain = nn.Sequential(
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            self.remain = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.main(x)
        # out = self.batchnorm(out)
        x = self.remain(x)
        return x

class LargeConvBlock(nn.Module):
    """
        Specific convolutional block followed by leakyrelu for unet.

                       Note:in_channels MUST == out_channels
        """

    def __init__(self, in_channels, out_channels, stride=1,parallel_sizes=None,mode='origin',batchnorm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.parallel_convs = nn.ModuleList()
        for kernel_size in parallel_sizes:
            padding = kernel_size // 2
            if mode == 'origin':
                self.parallel_convs.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding))
            elif mode == 'dw':
                self.parallel_convs.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding,groups=in_channels,bias=False))

        if batchnorm == True:
            self.remain = nn.Sequential(
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            self.remain = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = x
        for i, conv in enumerate(self.parallel_convs):
            y = y + conv(x)

        y = self.remain(y)
        return y


class LargeGhostBlock(nn.Module):
    def __init__(self,in_channels, out_channels, parallel_sizes, ratio=2):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvBlock(in_channels,init_channels,batchnorm = True)
        assert init_channels == new_channels, 'mid channel need == out channel in LargeGhostBlock!'
        self.cheap_operation = LargeConvBlock(
            in_channels=init_channels,
            out_channels=new_channels,
            parallel_sizes=parallel_sizes,
            mode='dw',
            batchnorm=True
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :, :]


class SpatialTransform(nn.Module):
    def __init__(self, device='cuda'):
        super(SpatialTransform, self).__init__()
        self.device = device

    def forward(self, mov_image, flow, mod='bilinear'):
        d2, h2, w2 = mov_image.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid(
            [torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(self.device).float()
        grid_d = grid_d.to(self.device).float()
        grid_w = grid_w.to(self.device).float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_d = flow[:, :, :, :, 0]
        flow_h = flow[:, :, :, :, 1]
        flow_w = flow[:, :, :, :, 2]

        # Remove Channel Dimension
        disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode=mod, align_corners=True)

        return warped


class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7, device='cuda'):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step
        self.device = device

    def forward(self, flow):
        # print(flow.shape)
        d2, h2, w2 = flow.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid(
            [torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(self.device).float()
        grid_d = grid_d.to(self.device).float()
        grid_w = grid_w.to(self.device).float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2 ** self.time_step)

        for i in range(self.time_step):
            flow_d = flow[:, 0, :, :, :]
            flow_h = flow[:, 1, :, :, :]
            flow_w = flow[:, 2, :, :, :]
            disp_d = (grid_d + flow_d).squeeze(1)
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)

            deformation = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
            flow = flow + torch.nn.functional.grid_sample(flow, deformation, mode='bilinear', padding_mode="border",
                                                          align_corners=True)
        return flow



