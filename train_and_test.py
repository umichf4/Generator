# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-17 15:20:26
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-09-04 19:52:27

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
from net import GeneratorNet
from utils import *
from tqdm import tqdm
from torch.optim import lr_scheduler


def train_generator(params):
    torch.set_default_tensor_type(torch.cuda.FloatTensor if params.cuda else torch.FloatTensor)
    # Device configuration
    device = torch.device('cuda:0' if params.cuda else 'cpu')
    print('Training starts, using %s' % (device))

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
    TT_pre, _ = load_mat(os.path.join(current_dir, params.T_path))
    TT_array, _ = data_pre(TT_pre, params.wlimit)

    np.random.shuffle(TT_array)
    all_num = TT_array.shape[0]
    TT_tensor = torch.from_numpy(TT_array).float()

    x = TT_tensor[:, 3:]
    train_x = x[:int(all_num * params.ratio), :]
    valid_x = x[int(all_num * params.ratio):, :]

    y = TT_tensor[:, :3]
    y[:, 0] = (y[:, 0] - 200) / 500
    y[:, 1] = (y[:, 1] - 20) / 70
    y[:, 2] = (y[:, 2] - 200) / 200
    train_y = y[:int(all_num * params.ratio)]
    valid_y = y[int(all_num * params.ratio):]

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_x, valid_y)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_x.shape[0], shuffle=True)

    # Net configuration
    net = GeneratorNet(in_num=params.in_num, out_num=params.out_num)

    optimizer = torch.optim.SGD(net.parameters(), lr=params.lr)
    scheduler = lr_scheduler.StepLR(optimizer, params.step_szie, params.gamma)

    criterion = nn.MSELoss()
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
            inputs, labels = data
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

            optimizer.zero_grad()
            net.zero_grad()

            outputs = net(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()

        train_loss_list.append(train_loss)

        # Validation
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                inputs, labels = data
                inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
                print(torch.max(labels))
                outputs = net(inputs)

                val_loss += criterion(outputs, labels)

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
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('matlab'))
    # Device configuration
    device = torch.device('cuda:0' if params.cuda else 'cpu')
    print('Test starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()

    _, TT_array = load_mat(os.path.join(current_dir, params.T_path))

    net = GeneratorNet(in_num=params.in_num, out_num=params.out_num)
    if params.restore_from:
        load_checkpoint(os.path.join(current_dir, params.restore_from), net, None)

    net.to(device)
    net.eval()

    spectrum_fake = []

    # with torch.no_grad():
    #     for thickness in range(300, 400, 100):
    #         for radius in range(60, 80, 10):
    #             for gap in range(200, 500, 100):

    #                 wavelength_real, spectrum_real = find_spectrum(thickness, radius, gap, TT_array)
    #                 input_tensor = torch.from_numpy(np.array(spectrum_real)).float().view(1, -1)
    #                 output_tensor = net(input_tensor.to(device))
    #                 device_param = np.array(output_tensor.view(-1).detach().cpu().numpy()).squeeze()
    #                 thickness_pred = device_param[0] * 500 + 200
    #                 radius_pred = device_param[1] * 70 + 20
    #                 gap_pred = device_param[2] * 200 + 200
    #                 title = 'thickness:{:.2f} radius:{:.2f} gap:{:.2f}'.format(thickness_pred, radius_pred, gap_pred)
    #                 print(title)
    #                 spectrum_fake = RCWA(eng, wavelength_real, thickness_pred, radius_pred, gap_pred)
    #                 plot_both_parts(wavelength_real, spectrum_real, spectrum_fake,
    #                                 params.shape + '_train_' + str(thickness) + '_' + str(radius) + '_' + str(gap) + '.png', legend=title)

    times = 5
    # valley_num = 4
    wavelength_real = np.linspace(400, 680, 29)

    with torch.no_grad():
        for i in range(times):

            # spectrum_real = random_gauss_spec_combo(wavelength_real, valley_num)
            spectrum_real = random_gauss_spec(wavelength_real)
            input_tensor = torch.from_numpy(np.array(spectrum_real)).float().view(1, -1)
            output_tensor = net(input_tensor.to(device))
            device_param = np.array(output_tensor.view(-1).detach().cpu().numpy()).squeeze()
            thickness_pred = device_param[0] * 500 + 200
            radius_pred = device_param[1] * 70 + 20
            gap_pred = device_param[2] * 200 + 200
            title = 'thickness:{:.2f} radius:{:.2f} gap:{:.2f}'.format(thickness_pred, radius_pred, gap_pred)
            print(title)
            spectrum_fake = RCWA(eng, wavelength_real, thickness_pred, radius_pred, gap_pred)
            plot_both_parts(wavelength_real, spectrum_real, spectrum_fake,
                            params.shape + '_test_' + str(i) + '.png', legend=title)

    print('Finished Testing \n')
