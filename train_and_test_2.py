# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-17 15:20:26
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-09-07 01:07:27

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import visdom
import numpy as np
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
from torch.utils.data import DataLoader, TensorDataset
from net.ArbitraryShape import GeneratorNet, SimulatorNet
from utils import *
from tqdm import tqdm
import cv2
import pytorch_ssim


def train_generator(params):
    torch.set_default_tensor_type(torch.cuda.FloatTensor if params.cuda else torch.FloatTensor)
    # type_tensor = torch.cuda.FloatTensor if params.cuda else torch.FloatTensor
    # Device configuration
    device = torch.device('cuda:0' if params.cuda else 'cpu')
    print('Training GeneratorNet starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()
    viz = visdom.Visdom()
    cur_epoch_loss = None
    cur_epoch_loss_opts = {
        'title': 'Epoch Loss Trace',
        'xlabel': 'Epoch Number',
        'ylabel': 'Loss',
        'width': 1200,
        'height': 600,
        'showlegend': True,
    }

    # Data configuration
    print("Waiting for data preparation...")
    _, TT_array = load_mat(os.path.join(current_dir, params.T_path))
    np.random.shuffle(TT_array)

    all_num, _, all_gap, all_spec, all_shape = data_pre_arbitrary(TT_array)

    train_gap = all_gap[:int(all_num * params.ratio)]
    valid_gap = all_gap[int(all_num * params.ratio):]

    train_spec = all_spec[:int(all_num * params.ratio), :]
    valid_spec = all_spec[int(all_num * params.ratio):, :]

    train_shape = all_shape[:int(all_num * params.ratio), :, :, :]
    valid_shape = all_shape[int(all_num * params.ratio):, :, :, :]

    train_dataset = TensorDataset(train_spec, train_shape, train_gap)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_spec, valid_shape, valid_gap)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_spec.shape[0], shuffle=True)

    # Net configuration
    net = GeneratorNet(noise_dim=params.noise_dim, spec_dim=params.spec_dim, d=params.net_depth)
    net.weight_init(mean=0, std=0.02)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.step_szie, params.gamma)

    criterion_1 = nn.MSELoss()
    criterion_2 = pytorch_ssim.SSIM(window_size=11)
    train_loss_list, val_loss_list, epoch_list = [], [], []

    if params.restore_from:
        load_checkpoint(params.restore_from, net, optimizer)

    net.to(device)

    if params.freeze:
        for param in net.deconv_block.parameters():
            param.requires_grad = False

    # Start training
    for k in range(params.epochs):
        epoch = k + 1
        epoch_list.append(epoch)

        # Train
        net.train()
        for i, data in enumerate(train_loader):

            inputs, labels, gaps = data
            inputs, labels, gaps = inputs.to(device), labels.to(device), gaps.to(device)
            noise = torch.rand(inputs.shape[0], params.noise_dim)

            optimizer.zero_grad()
            net.zero_grad()

            output_shapes, output_gaps = net(noise, inputs)
            train_loss = (params.beta * criterion_1(output_gaps, gaps) - criterion_2(output_shapes, labels))
            train_loss.backward()
            optimizer.step()

        train_loss_list.append(train_loss)

        # Validation
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                inputs, labels, gaps = data
                inputs, labels, gaps = inputs.to(device), labels.to(device), gaps.to(device)
                noise = torch.rand(inputs.shape[0], params.noise_dim)

                output_shapes, output_gaps = net(noise, inputs)
                val_loss += (params.beta * criterion_1(output_gaps, gaps) - criterion_2(output_shapes, labels))

        val_loss /= (i + 1)
        val_loss_list.append(val_loss)

        print('Epoch=%d train_loss: %.7f val_loss: %.7f lr: %.7f' %
              (epoch, train_loss, val_loss, scheduler.get_lr()[0]))

        scheduler.step()

        # Update Visualization
        if viz.check_connection():
            cur_epoch_loss = viz.line(torch.Tensor(train_loss_list), torch.Tensor(epoch_list),
                                      win=cur_epoch_loss, name='Train Loss',
                                      update=(None if cur_epoch_loss is None else 'replace'),
                                      opts=cur_epoch_loss_opts)
            cur_epoch_loss = viz.line(torch.Tensor(val_loss_list), torch.Tensor(epoch_list),
                                      win=cur_epoch_loss, name='Validation Loss',
                                      update=(None if cur_epoch_loss is None else 'replace'),
                                      opts=cur_epoch_loss_opts)

        if epoch % params.save_epoch == 0 and epoch != params.epochs:
            path = os.path.join(current_dir, params.save_model_dir)
            name = os.path.join(current_dir, params.save_model_dir, params.shape + '_Epoch' + str(epoch) + '.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                            path=path, name=name)

        if epoch == params.epochs:
            path = os.path.join(current_dir, params.save_model_dir)
            name = os.path.join(current_dir, params.save_model_dir, params.shape + '_Epoch' + str(epoch) + '_final.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                            path=path, name=name)

    print('Finished Training')


def test_generator(params):
    import matlab.engine
    from image_process import MetaShape
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('matlab'))
    # Device configuration
    device = torch.device('cuda:0' if params.cuda else 'cpu')
    print('Test starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()

    _, TT_array = load_mat(os.path.join(current_dir, params.T_path))

    net = GeneratorNet(noise_dim=params.noise_dim, spec_dim=params.spec_dim, d=params.net_depth)
    if params.restore_from:
        load_checkpoint(os.path.join(current_dir, params.restore_from), net, None)

    net.to(device)
    net.eval()
    wavelength = np.linspace(400, 680, 29)

    with torch.no_grad():
        TE_spec = random_step_spec(wavelength)
        TM_spec = TE_spec
        real_spec = (TE_spec + TM_spec) / 2
        spec = np.concatenate((TE_spec, TM_spec), axis=0)
        # spec = TT_array[0, 2:]
        # real_spec = (spec[: 29] + spec[29:]) / 2
        spec = torch.from_numpy(spec).float().view(1, -1)
        noise = torch.rand(1, params.noise_dim)
        spec, noise = spec.to(device), noise.to(device)
        output_img, ouput_gap = net(noise, spec)
        out_img = output_img.view(64, 64).detach().cpu().numpy()
        out_gap = int(ouput_gap.view(-1).detach().cpu().numpy() * 200 + 200)
        print(out_gap)
        shape_pred = MetaShape(out_gap)
        shape_pred.img = np.uint8(out_img * 255)
        shape_pred.binary_polygon()
        shape_pred.remove_small_twice()
        # shape_pred.erode_dilate(struc=3, iterations=2, mode="open")
        shape_pred.pad_boundary()
        # shape_pred.show_polygon(time=2000)
        shape_pred.save_polygon("hhhh.png")
        spec_pred_TE, spec_pred_TM = RCWA_arbitrary(eng, gap=out_gap, img_path="hhhh.png")
        fake_spec = (np.array(spec_pred_TE) + np.array(spec_pred_TM)) / 2
        plot_both_parts(wavelength, real_spec, fake_spec.squeeze(), "hhhh_result.png")

    print('Finished Testing \n')


def diff_tensor(a):
    a_new_right = torch.ones([a.shape[0], a.shape[1] + 1])
    a_new_right[:, 1:] = a
    a_new_left = torch.ones([a.shape[0], a.shape[1] + 1])
    a_new_left[:, :-1] = a
    a_diff = a_new_left - a_new_right
    a_diff = a_diff[:, 1:-1]
    return a_diff


def train_simulator(params):
    torch.set_default_tensor_type(torch.cuda.FloatTensor if params.cuda else torch.FloatTensor)
    # type_tensor = torch.cuda.FloatTensor if params.cuda else torch.FloatTensor
    # Device configuration
    device = torch.device('cuda:0' if params.cuda else 'cpu')
    print('Training SimulatorNet starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()
    viz = visdom.Visdom()
    cur_epoch_loss = None
    cur_epoch_loss_opts = {
        'title': 'Epoch Loss Trace',
        'xlabel': 'Epoch Number',
        'ylabel': 'Loss',
        'width': 1200,
        'height': 600,
        'showlegend': True,
    }

    # Data configuration
    print("Waiting for data preparation...")
    _, TT_array = load_mat(os.path.join(current_dir, params.T_path))
    np.random.shuffle(TT_array)

    all_num, _, all_gap, all_spec, all_shape = data_pre_arbitrary(TT_array)

    train_gap = all_gap[:int(all_num * params.ratio)]
    valid_gap = all_gap[int(all_num * params.ratio):]

    train_spec = all_spec[:int(all_num * params.ratio), :]
    valid_spec = all_spec[int(all_num * params.ratio):, :]

    train_shape = all_shape[:int(all_num * params.ratio), :, :, :]
    valid_shape = all_shape[int(all_num * params.ratio):, :, :, :]

    train_dataset = TensorDataset(train_spec, train_shape, train_gap)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_spec, valid_shape, valid_gap)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_spec.shape[0], shuffle=True)

    # Net configuration
    net = SimulatorNet(spec_dim=params.spec_dim, d=params.net_depth)
    net.weight_init(mean=0, std=0.02)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.step_szie, params.gamma)

    criterion = nn.MSELoss()
    train_loss_list, val_loss_list, epoch_list = [], [], []

    if params.restore_from:
        load_checkpoint(params.restore_from, net, optimizer)

    net.to(device)

    # Start training
    for k in range(params.epochs):
        epoch = k + 1
        epoch_list.append(epoch)

        # Train
        net.train()
        for i, data in enumerate(train_loader):

            specs, shapes, gaps = data
            specs, shapes, gaps = specs.to(device), shapes.to(device), gaps.to(device)

            optimizer.zero_grad()
            net.zero_grad()

            outputs = net(shapes, gaps)
            train_loss = criterion(F.interpolate(outputs.view(-1, 1, params.spec_dim), 100, mode='linear'),
                                   F.interpolate(specs.view(-1, 1, params.spec_dim), 100, mode='linear')) + \
                criterion(F.interpolate(diff_tensor(outputs.squeeze(1)).view(-1, 1, params.spec_dim - 1), 100, mode='linear'),
                          F.interpolate(diff_tensor(specs.squeeze(1)).view(-1, 1, params.spec_dim - 1), 100, mode='linear'))

            train_loss.backward()
            optimizer.step()

        train_loss_list.append(train_loss)

        # Validation
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                specs, shapes, gaps = data
                specs, shapes, gaps = specs.to(device), shapes.to(device), gaps.to(device)

                outputs = net(shapes, gaps)
                val_loss += criterion(F.interpolate(outputs.view(-1, 1, params.spec_dim), 100, mode='linear'),
                                      F.interpolate(specs.view(-1, 1, params.spec_dim), 100, mode='linear')) + \
                    criterion(F.interpolate(diff_tensor(outputs.squeeze(1)).view(-1, 1, params.spec_dim - 1), 100, mode='linear'),
                              F.interpolate(diff_tensor(specs.squeeze(1)).view(-1, 1, params.spec_dim - 1), 100, mode='linear'))

        val_loss /= (i + 1)
        val_loss_list.append(val_loss)

        print('Epoch=%d train_loss: %.7f val_loss: %.7f lr: %.7f' %
              (epoch, train_loss, val_loss, scheduler.get_lr()[0]))

        scheduler.step()

        # Update Visualization
        if viz.check_connection():
            cur_epoch_loss = viz.line(torch.Tensor(train_loss_list), torch.Tensor(epoch_list),
                                      win=cur_epoch_loss, name='Train Loss',
                                      update=(None if cur_epoch_loss is None else 'replace'),
                                      opts=cur_epoch_loss_opts)
            cur_epoch_loss = viz.line(torch.Tensor(val_loss_list), torch.Tensor(epoch_list),
                                      win=cur_epoch_loss, name='Validation Loss',
                                      update=(None if cur_epoch_loss is None else 'replace'),
                                      opts=cur_epoch_loss_opts)

        if epoch % params.save_epoch == 0 and epoch != params.epochs:
            path = os.path.join(current_dir, params.save_model_dir)
            name = os.path.join(current_dir, params.save_model_dir, params.shape + '_Epoch' + str(epoch) + '.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                            path=path, name=name)

        if epoch == params.epochs:
            path = os.path.join(current_dir, params.save_model_dir)
            name = os.path.join(current_dir, params.save_model_dir, params.shape + '_Epoch' + str(epoch) + '_final.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                            path=path, name=name)

    print('Finished Training')


def test_simulator(params):
    import matlab.engine
    from image_process import MetaShape
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('matlab'))
    # Device configuration
    device = torch.device('cuda:0' if params.cuda else 'cpu')
    print('Test starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()

    _, TT_array = load_mat(os.path.join(current_dir, params.T_path))

    net = SimulatorNet(spec_dim=params.spec_dim, d=params.net_depth)
    if params.restore_from:
        load_checkpoint(os.path.join(current_dir, params.restore_from), net, None)

    net.to(device)
    net.eval()
    wavelength = np.linspace(400, 960, 58)

    with torch.no_grad():
        real_spec = TT_array[0, 2:]
        gap = TT_array[0, 1]
        img = cv2.imread("polygon/0_342.png", flags=cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img).float().view(1, 1, 64, 64)
        gap = torch.from_numpy(np.array(gap)).float().view(1, 1)

        output = net(img, gap)
        fake_spec = output.view(-1).detach().cpu().numpy()
        plot_both_parts(wavelength, real_spec, fake_spec, "hhhh_result.png")

    print('Finished Testing \n')
