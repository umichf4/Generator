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
    
    sample_times = params.sample_times
    lamda = 0.01
    #spec_dim = 56
    spec_dim = 28
    spec_x = np.linspace(400, 680, spec_dim)
    w_list = list(spec_x)
    #noise_dim = 200
    noise_dim = 100
    gap_low = 200
    gap_high = 400
    gap_range = gap_high - gap_low
    t_low = 100
    t_high = 700
    t_range = t_high - t_low
    r_low = 20
    r_high = 80
    r_range = r_high - r_low
    
    # Fix the random seed
#    seed = 123
#    torch.manual_seed(seed)
#    np.random.seed(seed)
    
    # Net configuration
    out_features = 3
    net = G_Deconv_Fc(in_features=spec_dim + noise_dim, out_features=out_features) #0+56+200=256, gap+thick+r=3
    net = net.float().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=params.lr, momentum=0.9)
    #scheduler = lr_scheduler.StepLR(optimizer, params.step_szie, params.gamma)

    criterion_spec = nn.L1Loss()
    #criterion_para = Paraloss()
    #criterion_final = FinalLoss()
    loss_total_list, loss_mse_list, loss_entro_list, epoch_list = [], [], [], []

#    if params.restore_from:
#        load_checkpoint(params.restore_from, net, optimizer)        
    #load_checkpoint(params.restore_from, net_s, None)
    
    # Start training
    for k in range(params.epochs):
#        if k == 20:
#            kkk = k + 1
#            pass
        
        epoch = k + 1
        epoch_list.append(epoch)
        
        # Train
        net.train()
        
        spec = np.ones((params.batch_size, spec_dim))
#        spec_min_index = spec.argmin(1)
#        spec_left = np.ones((params.batch_size))
#        spec_right = np.ones((params.batch_size))
#        for i in range(params.batch_size):
#            if spec_min_index[i] - params.batch_size//4 >= 0:
#                spec_left[i] = spec_min_index[i] - params.batch_size//4
#            else:
#                spec_left[i] = 0
#                
#            if spec_min_index[i] + params.batch_size//4 < spec_dim:
#                spec_right[i] = spec_min_index[i] + params.batch_size//4
#            else:
#                spec_right[i] = 0
                           
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
        
        mu, var = net(spec, noise)
        mu = mu.unsqueeze(2) 
        var = var.unsqueeze(2) 
        mu = mu.expand(mu.shape[0], mu.shape[1], sample_times)
        var = var.expand(var.shape[0], var.shape[1], sample_times)
        
        entropies = []
        log_probs = []
        paras = []
        for i in range(out_features):
            para, log_prob, entropy = select_para(mu[:, i, :], var[:, i, :])
            
            entropies.append(entropy)
            log_probs.append(log_prob)
            paras.append(para)
                
        gap = (paras[0] * gap_range + gap_low).cpu().detach().numpy()            
        thick = (paras[1] * t_range + t_low).cpu().detach().numpy()
        radius =  (paras[2] * r_range + r_low).cpu().detach().numpy()
        
        loss_mse = 0
        loss_entro = 0
        loss_total = 0
        
        for j in range(sample_times):
            real_spec = RCWA(eng, w_list, gap[:,j].tolist(), thick[:,j].tolist(), radius[:,j].tolist(), acc=1)  # 不交叉采样
            real_spec = torch.from_numpy(real_spec).to(device).float()
            loss_spec = criterion_spec(spec, real_spec) 
            
            loss_mse = loss_mse - (log_probs[0][:,j] * log_probs[1][:,j] * log_probs[2][:,j] * log_probs[0][:,j].exp() * log_probs[1][:,j].exp() * log_probs[2][:,j].exp() * loss_spec).mean().to(device)
            loss_entro = loss_entro + (lamda * (entropies[0][:,j] + entropies[1][:,j] + entropies[2][:,j])).mean().to(device)
            #loss = loss + (log_probs[0][:,j] * log_probs[1][:,j] * log_probs[2][:,j] * loss_spec).sum().to(device) + (lamda * entropies[0][:,j] * entropies[1][:,j] * entropies[2][:,j]).sum().to(device)
            loss_total = loss_mse + loss_entro
        
        loss_mse = loss_mse / sample_times
        loss_entro = loss_entro / sample_times
        loss_total = loss_total / sample_times
        
        loss_total.backward()
        nn.utils.clip_grad_norm(net.parameters(), 50) # 梯度裁剪，防止梯度爆炸
        
        loss_mse_list.append(loss_mse)
        loss_entro_list.append(loss_entro)
        loss_total_list.append(loss_total)
        #loss_spec.requires_grad = True
        #loss_para = criterion_para(outputs.to(device))
        #loss_spec = Tensor(loss_spec.cpu().numpy())= 
        
        '''
        loss = criterion_final(loss_spec, outputs)
        loss.backward()
        loss_list.append(loss)
        '''
#        outputs = outputs * params.batch_size
#        loss_temp = criterion_spec(outputs, outputs * 2) * 100
#        loss_temp.backward()
        
        '''
        for i in net.named_parameters():
            print(i)
            break
        '''
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
        print('Epoch=%d  MSE Loss: %.7f  Entropy Loss: %.7f  Total Loss: %.7f ' %
              (epoch, loss_mse, loss_entro, loss_total))
        
#        plot_both(labels[0], inputs[0])
#        print(outputs[0])
        
        #scheduler.step()
        
        # Update Visualization
        if viz.check_connection():
            cur_epoch_loss = viz.line(torch.Tensor(loss_mse_list), torch.Tensor(epoch_list),
                                      win=cur_epoch_loss, name='MSE Loss',
                                      update=(None if cur_epoch_loss is None else 'replace'),
                                      opts=cur_epoch_loss_opts)
            cur_epoch_loss = viz.line(torch.Tensor(loss_entro_list), torch.Tensor(epoch_list),
                                      win=cur_epoch_loss, name='Entropy Loss',
                                      update=(None if cur_epoch_loss is None else 'replace'),
                                      opts=cur_epoch_loss_opts)
            cur_epoch_loss = viz.line(torch.Tensor(loss_total_list), torch.Tensor(epoch_list),
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
    
    spec_dim = 28
    noise_dim = 100
    
    spec = np.ones((params.batch_size, spec_dim))
    spec_x = np.linspace(400, 680, spec_dim)
    w_list = list(spec_x)
    for i in range(params.batch_size):
        spec[i] = random_gauss_spec(spec_x)
        
    spec = torch.from_numpy(spec)
    spec = spec.to(device).float()
    
    noise= np.random.uniform(low=0, high=1, size=(params.batch_size, noise_dim))
    noise = torch.from_numpy(noise)
    noise = noise.to(device).float()
        
    # Visualization configuration
    make_figure_dir()

    net = G_Deconv_Fc(in_features=spec_dim + noise_dim, out_features=3)
    net = net.float()
    net.to(device)
    if params.restore_from:
        load_checkpoint(os.path.join(current_dir, params.restore_from), net, None)
    net.eval()
    
    outputs = net(spec, noise)
    outputs = (outputs + 1) / 2
    
    gap = (outputs[:, 0] * 200 + 200).cpu().detach().numpy()
    thick = (outputs[:, 1] * 600 + 100).cpu().detach().numpy()
    radius =  (outputs[:, 2] * 80 + 20).cpu().detach().numpy()
    
    real_spec = RCWA(eng, w_list, list(gap), list(thick), list(radius), acc=1)
    real_spec = torch.from_numpy(real_spec).to(device).float()
    
    spec = spec.cpu().detach().numpy()
    real_spec = real_spec.cpu().detach().numpy()
    
    w = np.linspace(400,680,spec_dim)
    for i in range(5):
        plot_both_parts(w, real_spec[i], spec[i], str(i)+'spec[0].png') 

#    for thickness in thickness_all:
#        for radius in radius_all:
#
#            wavelength_real, spectrum_real = find_spectrum(thickness, radius, TT_array)
#            # if len(wavelength_real) > params.wlimit:
#            #     wavelength_real = wavelength_real[0:params.wlimit]
#            #     spectrum_real = spectrum_real[0:params.wlimit]
#            
#            # for wavelength in wavelength_real:
#            #     test_data = [wavelength, thickness, radius]
#            #     input_tensor = torch.from_numpy(np.array(test_data)).float().view(1, -1)
#            #     output_tensor = net(input_tensor.to(device))
#            #     spectrum_fake.append(output_tensor.view(-1).detach().cpu().numpy())
#
#            test_data = [thickness, radius]
#            input_tensor = torch.from_numpy(np.array(test_data)).float().view(1, -1)
#            output_tensor = net(input_tensor.to(device))
#            spectrum_fake = np.array(output_tensor.view(-1).detach().cpu().numpy()).squeeze()
#            plot_both_parts(wavelength_real, spectrum_real, spectrum_fake, str(thickness) + '_' + str(radius) + '.png')
#            print('Single iteration finished \n')

    print('Finished Testing \n')

