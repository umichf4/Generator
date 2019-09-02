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
#from PeleeNet import PeleeNet
#from net_2 import SimulatorNet
from InfoGAN import G_FC_Conv_Fc, G_Deconv_Fc
from utils import *
from utils import FinalLoss
#from utils import Paraloss
from tqdm import tqdm
from torch.optim import lr_scheduler
from scipy import interpolate
import random
import matplotlib.pyplot as plt
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath('matlab'))

def train_InfoGAN(params):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
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
    
    #spec_dim = 56
    spec_dim = 28
    spec_x = np.linspace(400, 680, spec_dim)
    w_list = list(spec_x)
    #noise_dim = 200
    noise_dim = 100

    # Net configuration
    net = G_Deconv_Fc(in_features=spec_dim + noise_dim, out_features=3) #0+56+200=256, gap+thick+r=3
    net = net.float().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=params.lr, momentum=0.9)
    #scheduler = lr_scheduler.StepLR(optimizer, params.step_szie, params.gamma)

    criterion_spec = nn.L1Loss()
    #criterion_para = Paraloss()
    criterion_final = FinalLoss()
    loss_list, epoch_list = [], []

#    if params.restore_from:
#        load_checkpoint(params.restore_from, net, optimizer)        
    #load_checkpoint(params.restore_from, net_s, None)
    
    # Start training
    for k in range(params.epochs):
        epoch = k + 1
        epoch_list.append(epoch)
        
        # Train
        net.train()
        
        spec = np.ones((params.batch_size, spec_dim))
        for i in range(params.batch_size):
            spec[i] = random_gauss_spec(spec_x)
            
        spec = torch.from_numpy(spec)
        spec = spec.to(device).float()
#        plt.plot(spec_x, spec, label='1')
#        plt.legend()
#        plt.grid()
#        plt.show()      
        noise= np.random.uniform(low=0, high=1, size=(params.batch_size, noise_dim))
        noise = torch.from_numpy(noise)
        noise = noise.to(device).float()
        
        optimizer.zero_grad()
        
        outputs = net(spec, noise)
        outputs = (outputs + 1) / 2
        
        
        gap = (outputs[:, 0] * 200 + 200).cpu().detach().numpy()
        thick = (outputs[:, 1] * 600 + 100).cpu().detach().numpy()
        radius =  (outputs[:, 2] * 80 + 20).cpu().detach().numpy()
        
        real_spec = RCWA(eng, w_list, list(gap), list(thick), list(radius), acc=1)
        real_spec = torch.from_numpy(real_spec).to(device).float()
        loss_spec = criterion_spec(spec, real_spec) 
        #loss.requires_grad = True
        #loss_para = criterion_para(outputs.to(device))
        #loss_spec = Tensor(loss_spec.cpu().numpy())
        
        loss = criterion_final(loss_spec, outputs)
        loss.backward()
        loss_list.append(loss)
        
#        outputs = outputs * params.batch_size
#        loss_temp = criterion_spec(outputs, outputs * 2) * 100
#        loss_temp.backward()
        
        for i in net.named_parameters():
            print(i)
            break

#        # Validation
#        net.eval()
#        val_loss = 0
#        for i, data in enumerate(valid_loader):
#            inputs = data[0].unsqueeze(1) 
#            inputs_inter = inter(inputs, device)
#            
#            outputs = net(inputs_inter)
#            
#            labels = disc_net(outputs, net_s, device)
#            
#            inputs = inputs.squeeze(1)
#            val_loss += criterion(inputs.to(device), labels).sum()
#
#        val_loss /= (i + 1)
#        val_loss_list.append(val_loss)
        optimizer.step()
        print('Epoch=%d  loss: %.7f' %
              (epoch, loss))
        
#        plot_both(labels[0], inputs[0])
#        print(outputs[0])
        
        #scheduler.step()
        
        # Update Visualization
        if viz.check_connection():
#            cur_epoch_loss = viz.line(torch.Tensor(spec_loss_list), torch.Tensor(epoch_list),
#                                      win=cur_epoch_loss, name='Spec Loss',
#                                      update=(None if cur_epoch_loss is None else 'replace'),
#                                      opts=cur_epoch_loss_opts)
#            cur_epoch_loss = viz.line(torch.Tensor(para_loss_list), torch.Tensor(epoch_list),
#                                      win=cur_epoch_loss, name='Para Loss',
#                                      update=(None if cur_epoch_loss is None else 'replace'),
#                                      opts=cur_epoch_loss_opts)
            cur_epoch_loss = viz.line(torch.Tensor(loss_list), torch.Tensor(epoch_list),
                                      win=cur_epoch_loss, name='Total Loss',
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


def test_InfoGAN(params):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Test starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()

    _, TT_array = load_mat(os.path.join(current_dir, params.T_path))

    net = SimulatorNet(in_num=params.in_num, out_num=params.out_num)
    net = net.float()
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
            #     input_tensor = torch.from_numpy(np.array(test_data)).float().view(1, -1)
            #     output_tensor = net(input_tensor.to(device))
            #     spectrum_fake.append(output_tensor.view(-1).detach().cpu().numpy())

            test_data = [thickness, radius]
            input_tensor = torch.from_numpy(np.array(test_data)).float().view(1, -1)
            output_tensor = net(input_tensor.to(device))
            spectrum_fake = np.array(output_tensor.view(-1).detach().cpu().numpy()).squeeze()
            plot_both_parts(wavelength_real, spectrum_real, spectrum_fake, str(thickness) + '_' + str(radius) + '.png')
            print('Single iteration finished \n')

    print('Finished Testing \n')

