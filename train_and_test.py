# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-17 15:20:26
# @Last Modified by:   BrandonHanx
# @Last Modified time: 2019-08-20 18:20:15

import torch
import torch.nn as nn
import visdom
import numpy as np
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
from torch.utils.data import DataLoader, TensorDataset
from PeleeNet import PeleeNet
from net_2 import SimulatorNet
from utils import *
from tqdm import tqdm
from torch.optim import lr_scheduler
from scipy import interpolate
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath('matlab'))

def train_simulator(params):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    TT_tensor = torch.from_numpy(TT_array)
    TT_tensor = TT_tensor.double()

    x = TT_tensor
    train_x = x[:int(all_num * params.ratio), 2:]
    valid_x = x[int(all_num * params.ratio):, 2:]

    train_dataset = TensorDataset(train_x)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_x)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_x.shape[0], shuffle=True)
    
    w = list(range(400, 681, 10))
    gap = 360
    acc = 5
    
    # Net configuration
    net = PeleeNet(num_classes=2)
    net = net.double()
    net.to(device)
    
    net_s = SimulatorNet(in_num=params.in_num, out_num=params.out_num)
    net_s = net_s.double()
    net_s.to(device)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=params.lr, momentum=0.9)
    #scheduler = lr_scheduler.StepLR(optimizer, params.step_szie, params.gamma)

    criterion = nn.L1Loss()
    train_loss_list, val_loss_list, epoch_list = [], [], []

#    if params.restore_from:
#        load_checkpoint(params.restore_from, net, optimizer)        
    load_checkpoint(params.restore_from, net_s, None)
    
    # Start training
    for k in range(params.epochs):
        epoch = k + 1
        epoch_list.append(epoch)
        
        # Train
        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs = data[0].unsqueeze(1) 
            inputs_inter = inter(inputs, device)
            
            optimizer.zero_grad()

            outputs = net(inputs_inter)     
            
            optimizer.step()
            
            #labels = disc_net(outputs, net_s, device)
            labels = RCWA(eng = eng, w_list = w, gap = gap, thick_list = outputs.cpu().detach().numpy()[:, 0].tolist(), r_list = outputs.cpu().detach().numpy()[:, 1].tolist(), acc = acc)
            
            inputs = inputs.squeeze(1)
            train_loss = criterion(inputs, labels)
            train_loss.backward()


        train_loss_list.append(train_loss)

        # Validation
        net.eval()
        val_loss = 0
        for i, data in enumerate(valid_loader):
            inputs = data[0].unsqueeze(1) 
            inputs_inter = inter(inputs, device)
            
            outputs = net(inputs_inter)
            
            labels = disc_net(outputs, net_s, device)
            
            inputs = inputs.squeeze(1)
            val_loss += criterion(inputs.to(device), labels).sum()

        val_loss /= (i + 1)
        val_loss_list.append(val_loss)

        print('Epoch=%d  train_loss: %.7f valid_loss: %.7f ' %
              (epoch, train_loss, val_loss))
        
        plot_both(labels[0], inputs[0])
        print(outputs[0])
        
        #scheduler.step()
        
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
            name = os.path.join(current_dir, params.save_model_dir, 'Epoch' + str(epoch) + '.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                            path=path, name=name)

        if epoch == params.epochs:
            path = os.path.join(current_dir, params.save_model_dir)
            name = os.path.join(current_dir, params.save_model_dir, 'Epoch' + str(epoch) + '_final.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                            path=path, name=name)

    print('Finished Training')


def test_simulator(params):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Test starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()

    _, TT_array = load_mat(os.path.join(current_dir, params.T_path))

    net = SimulatorNet(in_num=params.in_num, out_num=params.out_num)
    net = net.double()
    net.to(device)
    if params.restore_from:
        load_checkpoint(os.path.join(current_dir, params.restore_from), net, None)
    net.eval()
    thickness_all = range(200, 800, 100)
    radius_all = range(20, 100, 10)
    spectrum_fake = []

    for thickness in thickness_all:
        for radius in radius_all:

            wavelength_real, spectrum_real = find_spectrum(thickness, radius, TT_array)
            # if len(wavelength_real) > params.wlimit:
            #     wavelength_real = wavelength_real[0:params.wlimit]
            #     spectrum_real = spectrum_real[0:params.wlimit]
            
            # for wavelength in wavelength_real:
            #     test_data = [wavelength, thickness, radius]
            #     input_tensor = torch.from_numpy(np.array(test_data)).double().view(1, -1)
            #     output_tensor = net(input_tensor.to(device))
            #     spectrum_fake.append(output_tensor.view(-1).detach().cpu().numpy())

            test_data = [thickness, radius]
            input_tensor = torch.from_numpy(np.array(test_data)).double().view(1, -1)
            output_tensor = net(input_tensor.to(device))
            spectrum_fake = np.array(output_tensor.view(-1).detach().cpu().numpy()).squeeze()
            plot_both_parts(wavelength_real, spectrum_real, spectrum_fake, str(thickness) + '_' + str(radius) + '.png')
            print('Single iteration finished \n')

    print('Finished Testing \n')
