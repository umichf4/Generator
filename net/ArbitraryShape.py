# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-09-04 14:27:09
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-09-07 12:10:57

import os
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) *\
        torch.exp(
        -torch.sum((xy_grid - mean)**2., dim=-1) /
        (2 * variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel.requires_grad_(False)


class GeneratorNet(nn.Module):
    def __init__(self, noise_dim=100, spec_dim=58, d=64, kernel_size=5):
        super().__init__()
        self.noise_dim = noise_dim
        self.spec_dim = spec_dim
        self.gaussian_kernel = get_gaussian_kernel(kernel_size)
        self.pad = (kernel_size - 1) // 2
        self.deconv_block_spec = nn.Sequential(
            nn.ConvTranspose2d(self.spec_dim, d * 4, 4, 1, 0),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
        )
        self.deconv_block_noise = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim, d * 4, 4, 1, 0),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
        )
        self.deconv_block_cat = nn.Sequential(
            # ------------------------------------------------------
            nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose2d(d * 2, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose2d(d, 1, 4, 2, 1),
            nn.Tanh()
        )
        self.fc_block_1 = nn.Sequential(
            nn.Linear(64 * 64, 64 * 16),
            nn.BatchNorm1d(64 * 16),
            nn.LeakyReLU(0.2),
            nn.Linear(64 * 16, 64 * 4),
            nn.BatchNorm1d(64 * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(64 * 4, spec_dim),
            nn.BatchNorm1d(spec_dim),
            nn.LeakyReLU(0.2)
        )
        self.fc_block_2 = nn.Sequential(
            nn.Linear(spec_dim, 1),
            nn.ReLU6()
        )
        self.short_cut = nn.Sequential(
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, noise_in, spec_in):
        noise = self.deconv_block_noise(noise_in.view(-1, self.noise_dim, 1, 1))
        spec = self.deconv_block_spec(spec_in.view(-1, self.spec_dim, 1, 1))
        net = torch.cat((noise, spec), 1)
        img = self.deconv_block_cat(net)
        # net = F.conv2d(net, self.gaussian_kernel, padding=self.pad)
        # gap_in = self.fc_block_1(img.view(img.size(0), -1)) + self.short_cut(spec_in.view(spec_in.size(0), -1))
        # gap = self.fc_block_2(gap_in)
        return (img + 1) / 2


class SimulatorNet(nn.Module):
    def __init__(self, spec_dim=58, d=64):
        super().__init__()
        self.spec_dim = spec_dim
        self.conv_block_shape = nn.Sequential(
            # ------------------------------------------------------
            nn.Conv2d(1, d, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.conv_block_gap = nn.Sequential(
            # ------------------------------------------------------
            nn.ReplicationPad2d((63, 0, 63, 0)),
            nn.Conv2d(1, d // 4, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.conv_block_cat = nn.Sequential(
            # ------------------------------------------------------
            nn.Conv2d(d, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 2, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 4, d * 8, 4, 2, 1),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 8, d * 16, 4, 1, 0),
            nn.Sigmoid()
        )
        self.fc_block = nn.Sequential(
            nn.Linear(d * 16, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, spec_dim),
            nn.Sigmoid()
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, shape_in, gap_in):
        shape = self.conv_block_shape(shape_in)
        # gap = self.conv_block_gap(gap_in.view(-1, 1, 1, 1))
        # net = torch.cat((shape, gap), 1)
        spec = self.conv_block_cat(shape)
        spec = self.fc_block(spec.view(spec.shape[0], -1))
        return spec


if __name__ == '__main__':
    import torchsummary

    # if torch.cuda.is_available():
    #     generator = GeneratorNet().cuda()
    # else:
    #     generator = GeneratorNet()

    # torchsummary.summary(generator, [tuple([100]), tuple([58])])

    if torch.cuda.is_available():
        simulator = SimulatorNet().cuda()
    else:
        simulator = SimulatorNet()

    torchsummary.summary(simulator, [(1, 64, 64), tuple([1])])
