# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-17 15:20:26
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-08-29 16:04:22
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import json
import matplotlib.pyplot as plt
import cmath
import numpy as np
from scipy import interpolate
import scipy.io as scio
import matlab.engine
import math

current_dir = os.path.abspath(os.path.dirname(__file__))
pi = Variable(torch.FloatTensor([math.pi])).cuda()

gap_low = 200
gap_high = 400
gap_range = gap_high - gap_low
t_low = 100
t_high = 700
t_range = t_high - t_low
r_low = 20
r_high = 80
r_range = r_high - r_low
    
class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

class Paraloss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        length = x.shape[0]
        
        gap = x[:, 0].unsqueeze(1)
        gap = torch.abs(gap - 200) + torch.abs(gap - 400) - 200
        
        t = x[:, 1].unsqueeze(1)
        t = torch.abs(t - 100) + torch.abs(t - 700) - 600
        
        r = x[:, 2].unsqueeze(1)
        r = torch.abs(r - 20) + torch.abs(r - 100) - 80
        
        loss = gap + t + r
        loss = 0.01 * loss
        return torch.mean(torch.sum(loss, 1))
    
class FinalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, mse, outputs):
        loss = mse * outputs
        loss = torch.sum(torch.sum(loss, 1))
        
        return loss
    
def save_checkpoint(state, path, name):
    if not os.path.exists(path):
        print("Checkpoint Directory does not exist! Making directory {}".format(path))
        os.mkdir(path)

    torch.save(state, name)

    print('Model saved')


def load_checkpoint(path, net, optimizer):
    if not os.path.exists(path):
        raise("File doesn't exist {}".format(path))

    if torch.cuda.is_available():
        state = torch.load(path, map_location='cuda:0')
    else:
        state = torch.load(path, map_location='cpu')
    net.load_state_dict(state['net_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optim_state_dict'])

    print('Model loaded')


def interploate(org_data, points=1000):
    org = np.linspace(400, 680, len(org_data))
    new = np.linspace(400, 680, points)
    inter_func = interpolate.interp1d(org, org_data, kind='cubic')
    return inter_func(new)


def make_figure_dir():
    os.makedirs(current_dir + '/figures/loss_curves', exist_ok=True)
    os.makedirs(current_dir + '/figures/test_output', exist_ok=True)


def plot_single_part(wavelength, spectrum, name, legend='spectrum', interpolate=True):
    save_dir = os.path.join(current_dir, 'figures/test_output', name)
    plt.figure()
    plt.plot(wavelength, spectrum, 'ob')
    plt.grid()
    if interpolate:
        new_spectrum = interploate(spectrum)
        new_wavelength = interploate(wavelength)
        plt.plot(new_wavelength, new_spectrum, '-b')
    plt.title(legend)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(legend)
    plt.savefig(save_dir)
    plt.close()


def plot_both_parts(wavelength, real, fake, name, interpolate=True):

    color_left = 'blue'
    color_right = 'red'
    save_dir = os.path.join(current_dir, 'figures/test_output', name)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Real', color=color_left)
    ax1.plot(wavelength, real, 'o', color=color_left, label='Real')
    if interpolate:
        new_real = interploate(real)
        new_wavelength = interploate(wavelength)
        ax1.plot(new_wavelength, new_real, color=color_left)
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='y', labelcolor=color_left)
    ax1.grid()
    plt.ylim((0, 1))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Fake', color=color_right)  # we already handled the x-label with ax1
    ax2.plot(wavelength, fake, 'o', color=color_right, label='Fake')
    if interpolate:
        new_fake = interploate(fake)
        ax2.plot(new_wavelength, new_fake, color=color_right)
    ax2.legend(loc='upper right')
    ax2.tick_params(axis='y', labelcolor=color_right)
    ax2.spines['left'].set_color(color_left)
    ax2.spines['right'].set_color(color_right)
    plt.ylim((0, 1))
    plt.title('Real and Fake')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_dir)


def rect2polar(real, imag):
    complex_number = complex(real, imag)
    return abs(complex_number), cmath.phase(complex_number)


def polar2rect(modu, phase):
    complex_number = cmath.rect(modu, phase)
    return complex_number.real, complex_number.imag


def rect2polar_parallel(real_que, imag_que):
    assert len(real_que) == len(imag_que), "Size mismatch"
    modu_que, phase_que = np.zeros(len(real_que)), np.zeros(len(real_que))
    for i, real, imag in zip(range(len(real_que)), real_que, imag_que):
        modu_que[i], phase_que[i] = rect2polar(real, imag)
    return modu_que, phase_que


def polar2rect_parallel(modu_que, phase_que):
    assert len(modu_que) == len(phase_que), "Size mismatch"
    real_que, imag_que = np.zeros(len(modu_que)), np.zeros(len(modu_que))
    for i, modu, phase in zip(range(len(modu_que)), modu_que, phase_que):
        real_que[i], imag_que[i] = polar2rect(modu, phase)
    return real_que, imag_que


def find_spectrum(thickness, radius, TT_array):
    rows, _ = TT_array.shape
    wavelength, spectrum = [], []
    for row in range(rows):
        if TT_array[row, 1] == thickness and TT_array[row, 2] == radius:
            wavelength.append(TT_array[row, 0])
            spectrum.append(TT_array[row, -1])
        else:
            continue
    wavelength = np.array(wavelength)
    spectrum = np.array(spectrum)
    index_order = np.argsort(wavelength)
    return wavelength[index_order], spectrum[index_order]


def load_mat(path):
    variables = scio.whosmat(path)
    target = variables[0][0]
    data = scio.loadmat(path)
    TT_array = data[target]
    TT_list = TT_array.tolist()
    return TT_list, TT_array


def data_pre(list_all, wlimit):
    dtype = [('wave_length', int), ('thickness', int), ('radius', int), ('efficiency', float)]
    values = [tuple(single_device) for single_device in list_all]
    array_temp = np.array(values, dtype)
    array_all = np.sort(array_temp, order=['thickness', 'radius', 'wave_length'])

    thickness_list = np.unique(array_all['thickness'])
    radius_list = np.unique(array_all['radius'])
    reformed = []

    for thickness in thickness_list:
        for radius in radius_list:
            pick_index = np.intersect1d(np.argwhere(array_all['radius'] == radius), np.argwhere(
                array_all['thickness'] == thickness))
            picked = array_all[pick_index]
            picked = np.sort(picked, order=['wave_length'])
            cur_ref = [thickness, radius]
            for picked_single in picked:
                cur_ref.append(picked_single[3])

            # if len(cur_ref) > wlimit + 2:
            #     cur_ref = cur_ref[0:wlimit + 2]

            reformed.append(cur_ref)

    return np.array(reformed), array_all


def inter(inputs, device):
    inputs_inter = torch.ones(inputs.shape[0], inputs.shape[1], 224)
    x = np.linspace(0, 223, num=inputs.shape[2])
    new_x = np.linspace(0, 223, num=224)

    for index_j, j in enumerate(inputs):
        for index_jj, jj in enumerate(j):
            y = jj
            f = interpolate.interp1d(x, y, kind='cubic')
            jj = f(new_x)
            inputs_inter[index_j, index_jj, :] = torch.from_numpy(jj)

    inputs_inter = inputs_inter.double().to(device)

    return inputs_inter


def disc_net(in_data, net, device):
    outputs = net(in_data)

    outputs = outputs.double().to(device)
    return outputs


def RCWA(eng, w_list, gap_list, thick_list, r_list, acc=5):
    batch_size = len(thick_list)
    spec = np.ones((batch_size, len(w_list)))
    acc = matlab.double([acc])
    for i in range(batch_size):
        thick = thick_list[i]        
        r = r_list[i]        
        gap = gap_list[i]        
        
        gap, thick, r = range_restrict(gap, thick, r)
        
        thick = matlab.double([thick])
        r = matlab.double([r])
        gap = matlab.double([gap])
        
        for index, w in enumerate(w_list):
            w = matlab.double([w])
            eff = eng.RCWA_solver(w, gap, thick, r, acc)
            if type(eff) != float:
                print(eff)
                print(eff.size)
                print(gap)
                print(thick)
                print(r)
                return spec
            spec[i, index] = eff

    return spec

def range_restrict(gap, thick, r):
    if gap > gap_high:
        new_gap = gap_high
    elif gap < gap_low:
        new_gap = gap_low
    else: 
        new_gap = gap
        
    if thick > t_high:
        new_thick = t_high
    elif thick < t_low:
        new_thick = t_low
    else: 
        new_thick = thick
        
    if r > r_high:
        new_r = r_high
    elif r < r_low:
        new_r = r_low
    else: 
        new_r = r
        
    return new_gap, new_thick, new_r 
        
def plot_both(y1, y2):
    x = range(y1.shape[0])
    y1 = y1.cpu().detach().numpy()
    y2 = y2.cpu().detach().numpy()
    plt.figure()
    plt.plot(x, y1, label='Real', color='#F08080')
    plt.plot(x, y2, label='Predict', color='#DB7093')
    plt.show()


def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)


def gauss_spec(f, mean, var, depth=0.2):
    return 1 - (1 - depth) * np.exp(-(f - mean) ** 2 / (2 * (var ** 2)))


def random_gauss_spec(f):
    depth = np.random.uniform(low=0.0, high=0.05)
    mean = np.random.uniform(low=400, high=600)
    var = np.random.uniform(low=10, high=50)
    return 1 - (1 - depth) * np.exp(-(f - mean) ** 2 / (2 * (var ** 2)))


def random_gauss_spec_combo(f, valley_num):
    spec = np.zeros(len(f))
    for i in range(valley_num):
        spec += random_gauss_spec(f)
    return spec / valley_num


def normal(x, mu, sigma_sq):
    exp_part = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    coe_part = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return exp_part * coe_part


def select_para(mu, var):
    eps = torch.randn(mu.size()) / 2.0
    # calculate the probability
    para = (mu + var.sqrt()*Variable(eps).cuda()).data
    prob = normal(para, mu, var)
    prob = prob / (torch.sum(prob, 1).unsqueeze(1).expand(prob.shape)) # 对同一batch的同一参数的不同采样进行归一化
    entropy = 0.5*((2*(pi.expand_as(var))*var*(math.e)).log())

    log_prob = prob.log()
    return para, log_prob, entropy
    
def diff_tensor(a):
    a_new_right = torch.ones([a.shape[0], a.shape[1] + 1])
    a_new_right[:, 1:] = a
    a_new_left = torch.ones([a.shape[0], a.shape[1] + 1])
    a_new_left[:, :-1] = a
    a_diff = a_new_left - a_new_right
    a_diff = a_diff[:, 1:-1]
    return a_diff

if __name__ == "__main__":

    data_path = current_dir + '\\data'
    save_path = current_dir + '\\sorted'
    files = os.listdir(data_path)

    data_list_all = []

    for file in files:

        path = os.path.join(data_path, file)
        data_list, _ = load_mat(path)
        data_list_all.extend(data_list)

    data_prepared, data_sorted = data_pre(data_list_all, None)
    scio.savemat(save_path, {'value': data_prepared})

    # pick_index = np.intersect1d(np.argwhere(data_sorted['radius'] == 20), np.argwhere(data_sorted['thickness'] == 200))

    # picked_data = aray[pick_index]

    # sliced = array_slice(data_sorted)

    print('done')
