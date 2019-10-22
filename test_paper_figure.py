'''
Author: Luciano Hoult
Contact: lucianolee@zju.edu.cn
Company: Zhejiang University
Created Date: Sunday October 20th 2019 3:53:30 pm
Last Modified: Tuesday October 22nd 2019 2:11:48 pm
'''


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
from image_process import MetaShape


def test_paper(params):

    # Data configuration
    all_spec = np.load('data/all_spec.npy')
    print('Initial spectrum data loaded.')

    # Engine configuration
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('matlab'))
    eng.addpath(eng.genpath('solvers'))

    # Device configuration
    device = torch.device('cuda:0' if params.cuda else 'cpu')
    print('Test starts, using %s' % (device))
    
    c_spec_mse_all = []
    c_cv_mse_all = []
    d_spec_mse_all = []
    d_cv_mse_all = []
    e_spec_mse_all = []
    e_cv_mse_all = []
    f_spec_mse_all = []
    f_cv_mse_all = []
    
    # Mean of Guassian function shifts
    for g_mean in range(405, 685, 10):

        print('Testing of Gaussian with mean = %d starts.' % g_mean)

        wavelength = np.linspace(400, 680, 29)
        target_spec = self_gauss(mean=g_mean)
        target_cv = cal_contrast_vector(target_spec)
        net_ckp = 'models/generator_noctrast.pth'
        f_path = os.path.join(current_dir, "figures/test_output/" + str(g_mean))
        if not os.path.exists(f_path):
            os.mkdir(f_path)
        critic = 'MSE'
        
        # First generate figure c
        print('Generating figure c starts: searching for most fitting spectrum in database.')
        option = 'c'
        spec_mse, cv_mse = test_figure(params, option, eng, device, net_ckp, f_path,
                                       wavelength, critic, target_spec, target_cv, all_spec)
        c_spec_mse_all.append(spec_mse)
        c_cv_mse_all.append(cv_mse)
        print('Generating figure c Done')

        # Then generate figure d
        print('Generating figure d starts: input a spectrum to the net.')
        option = 'd'
        spec_mse, cv_mse = test_figure(params, option, eng, device, net_ckp, f_path,
                                       wavelength, critic, target_spec, target_cv, all_spec)
        d_spec_mse_all.append(spec_mse)
        d_cv_mse_all.append(cv_mse)
        print('Generating figure d Done')

        # Then generate figure e
        print('Generating figure e starts: input a cv to the net.')
        option = 'e'
        net_ckp = 'models/generator_full_trained.pth'
        spec_mse, cv_mse = test_figure(params, option, eng, device, net_ckp, f_path,
                                       wavelength, critic, target_spec, target_cv, all_spec)
        e_spec_mse_all.append(spec_mse)
        e_cv_mse_all.append(cv_mse)
        print('Generating figure e Done')

        # Then generate figure f
        print('Generating figure f starts: sort out the best.')
        option = 'f'
        net_ckp = 'models/generator_full_trained.pth'
        spec_mse, cv_mse = test_figure(params, option, eng, device, net_ckp, f_path,
                                       wavelength, critic, target_spec, target_cv, all_spec)
        f_spec_mse_all.append(spec_mse)
        f_cv_mse_all.append(cv_mse)
        print('Generating figure f Done')

        print('Testing of Gaussian with mean = %d finished.' % g_mean)

    mse_all = np.array([c_spec_mse_all, d_spec_mse_all, e_spec_mse_all, f_spec_mse_all,
                        c_cv_mse_all, d_cv_mse_all, e_cv_mse_all, f_cv_mse_all])

    np.save('figures/test_output/mse_all.npy', mse_all)

    plot_comparison(c_spec_mse_all, c_cv_mse_all, d_spec_mse_all, d_cv_mse_all,
                    e_spec_mse_all, e_cv_mse_all, f_spec_mse_all, f_cv_mse_all)

    print('All testing done.')


def test_figure(params, option, eng, device, net_ckp, f_path,
                wavelength, critic, target_spec, target_cv, all_spec):

    min_spec_mse = 1000
    min_cv_mse = 1000
    
    if option == 'c':

        min_spec_index = 0
        min_cv_index = 0

        with tqdm(total=5000) as t:
            for i in range(5000):

                cur_spec = all_spec[i, :29]
                cur_cv = cal_contrast_vector(cur_spec)
                cur_spec_mse = last_critic(cur_spec, target_spec)
                cur_cv_mse = last_critic(cur_cv, target_cv)
            
                if cur_spec_mse < min_spec_mse:
                    min_spec_mse = cur_spec_mse
                    min_spec_index = i

                if cur_cv_mse < min_cv_mse:
                    min_cv_mse = cur_cv_mse
                    min_cv_index = i

                t.update()

        min_spec_spec = all_spec[min_spec_index, :29]
        min_spec_cv = all_spec[min_cv_index, :29]
    
    elif option == 'd' or option == 'e' or option == 'f':

        target_spec_dual = np.concatenate((target_spec, target_spec))
        target_cv_dual = np.concatenate((target_cv, target_cv))
        iteration_num = 1

        min_spec_spec = np.array([])
        min_spec_cv = np.array([])

        if option == 'd':
            net = GeneratorNet(noise_dim=params.noise_dim, ctrast_dim=58, d=params.net_depth)
            net_in = torch.from_numpy(target_spec_dual).float().view(1, -1).to(device)
        else:
            net = GeneratorNet(noise_dim=params.noise_dim, ctrast_dim=14, d=params.net_depth)
            if option == 'e':
                net_in = torch.from_numpy(target_cv_dual).float().view(1, -1).to(device)
            else:
                iteration_num = 8

        load_checkpoint(net_ckp, net, None)
        net.to(device)
        net.eval()

        with torch.no_grad():

            noise = torch.rand(1, params.noise_dim).to(device)

            with tqdm(total=iteration_num) as t:
                for index in range(iteration_num):

                    if option == 'f':
                        tcvd = shuffle_tcvd(target_cv)
                        net_in = torch.from_numpy(tcvd).float().view(1, -1).to(device)

                    output_img, output_gap = net(noise, net_in)

                    print('\nFinishied single prediting.')

                    out_img = output_img.view(64, 64).detach().cpu().numpy()
                    out_gap = int(np.rint(output_gap.detach().cpu().numpy() * 200 + 200))

                    shape_pred = MetaShape(out_gap)
                    shape_pred.img = np.uint8(out_img * 255)
                    shape_pred.binary_polygon()
                    shape_pred.pad_boundary()
                    shape_pred.remove_small_twice()
                    polygon_dir = os.path.join(f_path, 'shape_' + option + str(iteration_num) + '.png')
                    shape_pred.save_polygon(polygon_dir)

                    spec_pred_TE, spec_pred_TM = RCWA_arbitrary(eng, gap=out_gap, img_path=polygon_dir)
                    fake_TM = np.array(spec_pred_TM).squeeze()
                    fake_TE = np.array(spec_pred_TE).squeeze()
                    fake_cv_TM = cal_contrast_vector(fake_TM)     
                    # both_spec = np.concatenate((fake_TM, fake_TE))
                    print('Finishied single RCWA.')

                    cur_spec_mse = last_critic(target_spec, fake_TM)
                    cur_cv_mse = last_critic(target_cv, fake_cv_TM)

                    if cur_spec_mse < min_spec_mse:
                        min_spec_mse = cur_spec_mse
                        min_spec_spec = fake_TM

                    if cur_cv_mse < min_cv_mse:
                        min_cv_mse = cur_cv_mse
                        min_spec_cv = fake_TM

                    t.update()

    else:
        raise Exception("Invalid operation.")

    print('Min error of spectrum is: %.7f' % (min_spec_mse))
    print('Min error of contrast vector is: %.7f' % (min_cv_mse))

    both_spec = np.concatenate((min_spec_spec, min_spec_cv))
    np_dir = os.path.join(f_path, option + '_ss_scv.npy')
    np.save(np_dir, both_spec)

    name = os.path.join(f_path, option + '_cv_comparison.png')
    if option == 'c' or option == 'f':
        plot_multiple_parts(wavelength, target_spec, min_spec_spec, fake_2=min_spec_cv, name=name, option_num=3)
    else:
        plot_multiple_parts(wavelength, target_spec, min_spec_spec, name=name, option_num=2)

    return min_spec_mse, min_cv_mse


def plot_comparison(c_spec_mse_all, c_cv_mse_all, d_spec_mse_all, d_cv_mse_all,
                    e_spec_mse_all, e_cv_mse_all, f_spec_mse_all, f_cv_mse_all):

    figure_1 = plt.figure(figsize=[10, 5])
    xaxis = np.linspace(405, 675, 28)
    plt.plot(xaxis, c_spec_mse_all, 'o-', color='lightsteelblue', label='c_spec_mse', linewidth=2.5)
    plt.plot(xaxis, d_spec_mse_all, 'o-', color='pink', label='d_spec_mse', linewidth=2.5)
    plt.plot(xaxis, e_spec_mse_all, 'o-', color='silver', label='e_spec_mse', linewidth=2.5)
    plt.plot(xaxis, f_spec_mse_all, 'o-', color='mediumseagreen', label='f_spec_mse', linewidth=2.5)
    plt.xlabel('Central position of Gaussian function (nm)')
    plt.ylabel('Spectrum MSE')
    plt.grid(linewidth=.5)
    plt.legend()
    save_dir = os.path.join(current_dir, 'figures/test_output/spec_comparison.jpg')
    plt.savefig(save_dir, dpi=600)

    figure_2 = plt.figure(figsize=[10, 5])
    plt.plot(xaxis, c_cv_mse_all, 'o-', color='lightsteelblue', label='c_cv_mse', linewidth=2.5)
    plt.plot(xaxis, d_cv_mse_all, 'o-', color='pink', label='d_cv_mse', linewidth=2.5)
    plt.plot(xaxis, e_cv_mse_all, 'o-', color='silver', label='e_cv_mse', linewidth=2.5)
    plt.plot(xaxis, f_cv_mse_all, 'o-', color='mediumseagreen', label='f_cv_mse', linewidth=2.5)
    plt.xlabel('Central position of Gaussian function (nm)')
    plt.ylabel('Contrast Vector MSE')
    plt.grid(linewidth=.5)
    plt.legend()
    save_dir = os.path.join(current_dir, 'figures/test_output/cv_comparison.jpg')
    plt.savefig(save_dir, dpi=600)


def plot_multiple_parts(wavelength, real, fake_1, fake_2=None,
                        name='figures/cv_com.jpg', option_num=2, interpolate=True,
                        legend='Target & Predicted Spectra and Corresponding Contrast Vectors'):

    cv1 = cal_contrast_vector(real)
    cv2 = cal_contrast_vector(fake_1)
    if option_num == 3:
        cv3 = cal_contrast_vector(fake_2)

    fig, ax1 = plt.subplots(figsize=[7, 5])
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Transimttance', color='blue')
    ax1.plot(wavelength, real, 'o', color='mediumseagreen', label='Target Spectrum')
    ax1.plot(interploate(wavelength), interploate(real), color='mediumseagreen')
    ax1.plot(wavelength, fake_1, 'o', color='darkturquoise', label='Predicted Spectrum (by min_spec_mse)')
    ax1.plot(interploate(wavelength), interploate(fake_1), color='darkturquoise')
    if option_num == 3:
        ax1.plot(wavelength, fake_2, 'o', color='cornflowerblue', label='Predicted Spectrum (by min_cv_mse)')
        ax1.plot(interploate(wavelength), interploate(fake_2), color='cornflowerblue', label='Predicted Spectrum (by min_cv_mse)')

    # ax1.legend(loc='upper left')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid()
    plt.ylim((0, 1))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Contrast', color='red')  # we already handled the x-label with ax1
    ax2.step(np.linspace(400, 680, 8), np.append(cv1, cv1[-1]), where='post', color='lightcoral', label='Target Contrast Vector')
    ax2.step(np.linspace(400, 680, 8), np.append(cv2, cv2[-1]), where='post', color='violet', label='Predicted Contrast Vector (by min_spec_mse)')
    if option_num == 3:
        ax2.step(np.linspace(400, 680, 8), np.append(cv3, cv3[-1]), where='post', color='darkorange', label='Predicted Contrast Vector (by min_spec_mse)')    
    # ax2.legend(loc='upper right')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.spines['left'].set_color('blue')
    ax2.spines['right'].set_color('red')
    plt.title(legend)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(name, dpi=600)
    
    # save_dir_eps = os.path.join(current_dir, 'figures/test_output', name[:-4] + '_eps_ver')
    # plt.savefig(save_dir_eps, format="eps")
    # plt.show()


def last_critic(vector_1, vector_2, option='mse'):
    
    if option == 'mse':
        return np.square(np.subtract(vector_1, vector_2)).mean()

    else:
        pass


def self_gauss(mean=460, variance=25, amplitude=0.9, up=0):
    
    wave_length = np.linspace(400, 680, 29)
    
    if up:
        return amplitude * np.exp(-np.power(wave_length - mean, 2.) / (2 * np.power(variance, 2.)))

    else:
        return 1 - amplitude * np.exp(-np.power(wave_length - mean, 2.) / (2 * np.power(variance, 2.)))


def shuffle_tcvd(target_cv):

    min_index = np.argmin(target_cv)
    min_val = target_cv[min_index]
    
    cv_all = np.zeros(7)
    for index in range(7):
        choice_other = np.random.choice([0.8, 0.9, 1.0])
        cv_all[index] = choice_other

    cv_all[min_index] = min_val
    cv_dual = np.concatenate((cv_all, cv_all))

    return cv_dual


if __name__ == '__main__':
    
    # da_path = os.path.join(current_dir, 'data/shape_spec_6500.mat')
    # data_pre_arbitrary(da_path)
    mse_all = np.load(os.path.join(current_dir, 'figures/test_output/mse_all.npy'))
    c_spec_mse_all = mse_all[0]
    d_spec_mse_all = mse_all[1]
    e_spec_mse_all = mse_all[2]
    f_spec_mse_all = mse_all[3]
    c_cv_mse_all = mse_all[4]
    d_cv_mse_all = mse_all[5]
    e_cv_mse_all = mse_all[6]
    f_cv_mse_all = mse_all[7]

    plot_comparison(c_spec_mse_all, c_cv_mse_all, d_spec_mse_all, d_cv_mse_all,
                    e_spec_mse_all, e_cv_mse_all, f_spec_mse_all, f_cv_mse_all)

    plt.show()