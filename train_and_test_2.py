# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-17 15:20:26
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-09-09 16:36:01

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
    if not (os.path.exists('data/all_ctrast.npy') and os.path.exists('data/all_gap.npy') and os.path.exists('data/all_shape.npy') and os.path.exists('data/all_spec.npy') and os.path.exists('data/all_gauss.npy')):
        _, _, all_gap_np, all_spec_np, all_shape_np, all_gauss_np, all_ctrast_np = data_pre_arbitrary(params.T_path)
        np.save('data/all_gap.npy', all_gap_np)
        np.save('data/all_spec.npy', all_spec_np)
        np.save('data/all_shape.npy', all_shape_np)
        np.save('data/all_ctrast.npy', all_ctrast_np)
        # np.save('data/all_gauss.npy', all_gauss_np)

    all_gap = torch.from_numpy(np.load('data/all_gap.npy')).float()
    all_spec = torch.from_numpy(np.load('data/all_spec.npy')).float()
    all_shape = torch.from_numpy(np.load('data/all_shape.npy')).float()
    all_ctrast = torch.from_numpy(np.load('data/all_ctrast.npy')).float()
    # all_gauss = torch.from_numpy(np.load('data/all_gauss.npy')).float()

    all_num = all_gap.shape[0]

    # train_gap = all_gap[:int(all_num * params.ratio)]
    # valid_gap = all_gap[int(all_num * params.ratio):]

    train_spec = all_spec[:int(all_num * params.ratio), :]
    valid_spec = all_spec[int(all_num * params.ratio):, :]

    train_ctrast = all_ctrast[:int(all_num * params.ratio), :]
    valid_ctrast = all_ctrast[int(all_num * params.ratio):, :]

    # train_gauss = all_gauss[:int(all_num * params.ratio), :]
    # valid_gauss = all_gauss[int(all_num * params.ratio):, :]

    train_shape = all_shape[:int(all_num * params.ratio), :, :, :]
    valid_shape = all_shape[int(all_num * params.ratio):, :, :, :]

    train_dataset = TensorDataset(train_spec, train_shape, train_ctrast)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_spec, valid_shape, valid_ctrast)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_spec.shape[0], shuffle=True)

    # Net configuration
    net = GeneratorNet(noise_dim=params.noise_dim, spec_dim=params.spec_dim, d=params.net_depth)
    net.weight_init(mean=0, std=0.02)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.step_szie, params.gamma)

    simulator = SimulatorNet(spec_dim=params.spec_dim, d=params.net_depth)
    load_checkpoint('models/arbitrary_Epoch1000_final_s.pth', simulator, None)
    for param in simulator.parameters():
        param.requires_grad = False

    criterion_1 = nn.MSELoss()
    criterion_2 = pytorch_ssim.SSIM(window_size=11)
    train_loss_list, val_loss_list, epoch_list = [], [], []
    spec_loss, shape_loss = 0, 0

    if params.restore_from:
        load_checkpoint(params.restore_from, net, optimizer)

    net.to(device)
    simulator.to(device)
    simulator.eval()

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

            inputs, labels, ctrasts = data
            inputs, labels, ctrasts = inputs.to(device), labels.to(device), ctrasts.to(device)
            noise = torch.rand(inputs.shape[0], params.noise_dim)

            optimizer.zero_grad()
            net.zero_grad()

            output_shapes, output_gaps = net(noise, ctrasts)
            output_specs = simulator(output_shapes, output_gaps)
            spec_loss = criterion_1(output_specs, inputs)
            shape_loss = 1 - criterion_2(output_shapes, labels)
            train_loss = spec_loss + shape_loss * 0.1
            train_loss.backward()
            optimizer.step()

        train_loss_list.append(train_loss)

        # Validation
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                inputs, labels, ctrasts = data
                inputs, labels, ctrasts = inputs.to(device), labels.to(device), ctrasts.to(device)
                noise = torch.rand(inputs.shape[0], params.noise_dim)

                output_shapes, output_gaps = net(noise, ctrasts)
                output_specs = simulator(output_shapes, output_gaps)
                spec_loss = criterion_1(output_specs, inputs)
                shape_loss = 1 - criterion_2(output_shapes, labels)
                val_loss += spec_loss + shape_loss * 0.1

        val_loss /= (i + 1)
        val_loss_list.append(val_loss)

        print('Epoch=%d train_loss: %.7f val_loss: %.7f spec_loss: %.7f shape_loss: %.7f lr: %.7f' %
              (epoch, train_loss, val_loss, spec_loss, shape_loss, scheduler.get_lr()[0]))
        # print('Epoch=%d train_loss: %.7f val_loss: %.7f lr: %.7f' %
        #       (epoch, train_loss, val_loss, scheduler.get_lr()[0]))

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
    eng.addpath(eng.genpath('solvers'))
    # Device configuration
    device = torch.device('cuda:0' if params.cuda else 'cpu')
    print('Test starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()

    net = GeneratorNet(noise_dim=params.noise_dim, spec_dim=params.spec_dim, d=params.net_depth)
    if params.restore_from:
        load_checkpoint(os.path.join(current_dir, params.restore_from), net, None)

    net.to(device)
    net.eval()
    wavelength = np.linspace(400, 680, 29)
    lucky = np.random.randint(low=int(4881 * params.ratio), high=4881)
    all_spec = np.load('data/all_spec.npy')
    # all_gap = np.load('data/all_gap.npy')
    # all_shape = np.load('data/all_shape.npy')

    with torch.no_grad():
        # real_spec = all_spec[int(lucky)]
        real_spec = gauss_spec_valley(wavelength, 440, 30, 0.1)
        spec = np.concatenate((real_spec, real_spec))
        spec = torch.from_numpy(spec).float().view(1, -1)
        noise = torch.rand(1, params.noise_dim)
        spec, noise = spec.to(device), noise.to(device)
        output_img, ouput_gap = net(noise, spec)
        out_img = output_img.view(64, 64).detach().cpu().numpy()
        out_gap = int(np.rint(ouput_gap.view(-1).detach().cpu().numpy() * 200 + 200))
        print(out_gap)
        shape_pred = MetaShape(out_gap)
        shape_pred.img = np.uint8(out_img * 255)
        shape_pred.binary_polygon()
        shape_pred.remove_small_twice()
        shape_pred.pad_boundary()
        shape_pred.save_polygon("figures/test_output/hhhh.png")

        spec_pred_TE, spec_pred_TM = RCWA_arbitrary(eng, gap=out_gap, img_path="figures/test_output/hhhh.png")
        fake_spec = np.array(spec_pred_TE)
        plot_both_parts(wavelength, real_spec[0:29], fake_spec.squeeze(), "hhhh_result.png")

    print('Finished Testing \n')


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
    if not (os.path.exists('data/all_gap.npy') and os.path.exists('data/all_shape.npy') and os.path.exists('data/all_spec.npy') and os.path.exists('data/all_gauss.npy')):
        _, _, all_gap_np, all_spec_np, all_shape_np, all_gauss_np = data_pre_arbitrary(params.T_path)
        np.save('data/all_gap.npy', all_gap_np)
        np.save('data/all_spec.npy', all_spec_np)
        np.save('data/all_shape.npy', all_shape_np)
        # np.save('data/all_gauss.npy', all_gauss_np)

    all_gap = torch.from_numpy(np.load('data/all_gap.npy')).float()
    all_spec = torch.from_numpy(np.load('data/all_spec.npy')).float()
    all_shape = torch.from_numpy(np.load('data/all_shape.npy')).float()
    # all_gauss = torch.from_numpy(np.load('data/all_gauss.npy')).float()

    all_num = all_gap.shape[0]

    train_gap = all_gap[:int(all_num * params.ratio)]
    valid_gap = all_gap[int(all_num * params.ratio):]

    train_spec = all_spec[:int(all_num * params.ratio), :]
    valid_spec = all_spec[int(all_num * params.ratio):, :]

    # train_gauss = all_gauss[:int(all_num * params.ratio), :]
    # valid_gauss = all_gauss[int(all_num * params.ratio):, :]

    train_shape = all_shape[:int(all_num * params.ratio), :, :, :]
    valid_shape = all_shape[int(all_num * params.ratio):, :, :, :]

    train_dataset = TensorDataset(train_spec, train_shape, train_gap)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_spec, valid_shape, valid_gap)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_gap.shape[0], shuffle=True)

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
            train_loss = criterion(outputs, specs)
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
                val_loss += criterion(outputs, specs)
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
    eng.addpath(eng.genpath('solvers'))
    # Device configuration
    device = torch.device('cuda:0' if params.cuda else 'cpu')
    print('Test starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()

    net = SimulatorNet(spec_dim=params.spec_dim, d=params.net_depth)
    if params.restore_from:
        load_checkpoint(os.path.join(current_dir, params.restore_from), net, None)

    net.to(device)
    net.eval()
    wavelength = np.linspace(400, 680, 29)
    lucky = np.random.randint(low=int(4881 * params.ratio), high=4881)
    all_spec = np.load('data/all_spec.npy')
    all_gap = np.load('data/all_gap.npy')
    all_shape = np.load('data/all_shape.npy')

    with torch.no_grad():
        real_spec = all_spec[int(lucky)]
        gap = all_gap[int(lucky)]
        img = all_shape[int(lucky)]
        # cv2.imwrite("hhh.png", img.reshape(64, 64) * 255)
        spec = torch.from_numpy(real_spec).float().view(1, -1)
        img = torch.from_numpy(img).float().view(1, 1, 64, 64)
        gap = torch.from_numpy(np.array(gap)).float().view(1, 1)

        output = net(img, gap)
        fake_spec = output.view(-1).detach().cpu().numpy()
        plot_both_parts(wavelength, real_spec[:29], fake_spec[:29], "hhhh_result.png")
        loss = F.mse_loss(output, spec)
        print(lucky, loss)

    print('Finished Testing \n')
