# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-17 15:20:26
# @Last Modified by:   BrandonHanx
# @Last Modified time: 2019-09-11 13:32:55
import torch
import os
import json
import matplotlib.pyplot as plt
import cmath
import numpy as np
from scipy import interpolate
import scipy.io as scio
import cv2
from tqdm import tqdm
from scipy.optimize import curve_fit
from imutils import rotate_bound
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.abspath(os.path.dirname(__file__))
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


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
    plt.ylim((0, 1))
    plt.savefig(save_dir)
    plt.close()


def plot_both_parts(wavelength, real, fake, name, legend='Real and Fake', interpolate=True):

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
    plt.title(legend)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_dir)


def plot_both_parts_2(wavelength, real, cv, name, legend='Spectrum and Contrast Vector', interpolate=True):

    color_left = 'blue'
    color_right = 'red'
    save_dir = os.path.join(current_dir, 'figures/test_output', name)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Transimttance', color=color_left)
    ax1.plot(wavelength, real, 'o', color=color_left, label='Spectrum')
    if interpolate:
        new_real = interploate(real)
        new_wavelength = interploate(wavelength)
        ax1.plot(new_wavelength, new_real, color=color_left)
    # ax1.legend()
    ax1.tick_params(axis='y', labelcolor=color_left)
    ax1.grid()
    plt.ylim((0, 1))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Contrast', color=color_right)  # we already handled the x-label with ax1
    ax2.step(np.linspace(400, 680, 8), np.append(cv, cv[-1]), where='post', color=color_right, label='Contrast Vector')
    # ax2.legend()
    ax2.tick_params(axis='y', labelcolor=color_right)
    ax2.spines['left'].set_color(color_left)
    ax2.spines['right'].set_color(color_right)
    # plt.ylim((0, 1))
    plt.title(legend)

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


def find_spectrum(thickness, radius, gap, TT_array):
    rows, _ = TT_array.shape
    wavelength, spectrum = [], []
    for row in range(rows):
        if TT_array[row, 1] == thickness and TT_array[row, 2] == radius and TT_array[row, 3] == gap:
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
    dtype = [('wave_length', int), ('thickness', int), ('radius', int), ('gap', int), ('efficiency', float)]
    values = [tuple(single_device) for single_device in list_all]
    array_temp = np.array(values, dtype)
    array_all = np.sort(array_temp, order=['thickness', 'radius', 'gap', 'wave_length'])

    thickness_list = np.unique(array_all['thickness'])
    radius_list = np.unique(array_all['radius'])
    gap_list = np.unique(array_all['gap'])
    reformed = []

    for thickness in thickness_list:
        for radius in radius_list:
            for gap in gap_list:
                pick_index = np.intersect1d(np.argwhere(array_all['radius'] == radius), np.argwhere(
                    array_all['thickness'] == thickness))
                pick_index = np.intersect1d(pick_index, np.argwhere(array_all['gap'] == gap))
                picked = array_all[pick_index]
                picked = np.sort(picked, order=['wave_length'])
                cur_ref = [thickness, radius, gap]
                for picked_single in picked:
                    cur_ref.append(picked_single[4])

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


def RCWA_parallel(eng, w_list, thick_list, r_list, gap, acc=5):
    import matlab.engine
    batch_size = len(thick_list)
    spec = np.ones((batch_size, len(w_list)))
    gap = matlab.double([gap])
    acc = matlab.double([acc])
    for i in range(batch_size):
        thick = thick_list[i]
        thick = matlab.double([thick])
        r = r_list[i]
        r = matlab.double([r])
        for index, w in enumerate(w_list):
            w = matlab.double([w])
            spec[i, index] = eng.RCWA_solver(w, gap, thick, r, acc)

    return spec


def RCWA(eng, w_list, thick, r, gap, acc=5, medium=1, shape=0):
    import matlab.engine
    spec = np.ones(len(w_list))
    gap = matlab.double([gap])
    acc = matlab.double([acc])
    thick = matlab.double([thick])
    r = matlab.double([r])
    medium = matlab.double([medium])
    shape = matlab.double([shape])
    for index, w in enumerate(w_list):
        w = matlab.double([w])
        spec[index] = eng.RCWA_solver(w, gap, thick, r, acc, medium, shape)

    return spec


def RCWA_arbitrary(eng, gap, img_path, thickness=500, acc=5):
    import matlab.engine
    gap = matlab.double([gap])
    acc = matlab.double([acc])
    thick = matlab.double([thickness])
    spec = eng.cal_spec(gap, thick, acc, img_path, nargout=2)
    spec_TE, spec_TM = spec
    return spec_TE, spec_TM


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def keep_range(data, low=0, high=1):
    index = np.where(data > high)
    data[index] = high
    index = np.where(data < low)
    data[index] = low

    return data


def gauss_spec_valley(f, mean, var, depth=0.2):
    return 1 - (1 - depth) * np.exp(-(f - mean) ** 2 / (2 * (var ** 2)))


def gauss_spec_peak(f, mean, var, depth=0.2):
    return (1 - depth) * np.exp(-(f - mean) ** 2 / (2 * (var ** 2)))


def random_gauss_spec(f):
    depth = np.random.uniform(low=0.0, high=0.05)
    mean = np.random.uniform(low=400, high=600)
    var = np.random.uniform(low=20, high=40)
    return 1 - (1 - depth) * np.exp(-(f - mean) ** 2 / (2 * (var ** 2)))


def random_step_spec(f):
    depth = np.random.uniform(low=0.0, high=0.2)
    duty_ratio = np.random.uniform(low=0.1, high=0.3)
    width = int(len(f) * duty_ratio / 2)
    valley = np.random.randint(low=len(f) / 3, high=2 * len(f) / 3, dtype=int)
    spec = np.random.uniform(low=0.7, high=1, size=len(f))
    spec[valley - width:valley + width] = depth
    return spec


def random_gauss_spec_combo(f, valley_num):
    spec = np.zeros(len(f))
    for i in range(valley_num):
        spec += random_gauss_spec(f)
    return normalization(spec)


def spec_jitter(spec, amp):
    return normalization(spec + np.random.uniform(low=-amp, high=amp, size=spec.size))


def gauss10(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9):

    return a0 * np.exp(-((x - m0) / s0)**2) + a1 * np.exp(-((x - m1) / s1)**2) + a2 * np.exp(-((x - m2) / s2)**2) + \
        a3 * np.exp(-((x - m3) / s3)**2) + a4 * np.exp(-((x - m4) / s4)**2) + a5 * np.exp(-((x - m5) / s5)**2) + \
        a6 * np.exp(-((x - m6) / s6)**2) + a7 * np.exp(-((x - m7) / s7)**2) + a8 * np.exp(-((x - m8) / s8)**2) + \
        a9 * np.exp(-((x - m9) / s9)**2)


def gauss10_tensor(in_tensor):
    wave_tensor = torch.range(400, 680, 10)

    def gauss_tensor(wave_tensor, a, m, s):
        return a * torch.exp(-torch.pow((wave_tensor - m) / s, 2))

    out_tensor = torch.zeros_like(wave_tensor)
    for i in range(10):
        out_tensor += gauss_tensor(wave_tensor, in_tensor[i], in_tensor[i + 10], in_tensor[i + 20])
    return out_tensor


def gauss10_curve_fit(spec):
    a_min = [0] * 10
    a_max = [1] * 10
    m_min = [400] * 10
    m_max = [680] * 10
    s_min = [0] * 10
    s_max = [1000] * 10
    min_limit = a_min + m_min + s_min
    max_limit = a_max + m_max + s_max
    wavelength = np.linspace(400, 680, 29)
    popt, _ = curve_fit(gauss10, wavelength, spec, bounds=(min_limit, max_limit))
    return popt


def cal_contrast(wavelength, spec, spec_start, spec_end):
    spec_range_in = spec[np.argwhere((wavelength <= spec_end) & (wavelength >= spec_start))]
    sepc_range_out = spec[np.argwhere((wavelength > spec_end) | (wavelength < spec_start))]
    contrast = np.max(spec_range_in) / np.max(sepc_range_out)
    return contrast


def cal_contrast_vector(spec):
    wavelength = np.linspace(400, 680, 29)
    contrast_vector = np.zeros(7)
    for i in range(len(contrast_vector)):
        contrast_vector[i] = cal_contrast(wavelength, spec, 400 + i * 40, 440 + i * 40)
    return contrast_vector


def plot_possible_spec(spec):
    min_index = np.argmin(spec, axis=1)
    min_sort = np.argsort(min_index)
    TE_spec = spec[min_sort, :]
    wavelength = np.linspace(400, 680, 29)
    # TE_spec = cv2.resize(src=TE_spec, dsize=(1000, 1881), interpolation=cv2.INTER_CUBIC)

    plt.figure()
    plt.pcolor(TE_spec, cmap=plt.cm.jet)
    plt.xlabel('Wavelength (nm)')
    # plt.xlabel('Index of elements')
    plt.ylabel('Index of Devices')
    plt.title('Possible Contrast Distribution (TM)')
    # plt.title('Gaussian Amplitude after Decomposition')
    # plt.title('Possible Spectrums of Arbitrary Shapes (' + title + ')')
    # plt.title(r'Possible Spectrums of Square Shape ($T_iO_2$)')
    # plt.xticks(np.arange(len(wavelength), step=4), np.uint16(wavelength[::4]))
    plt.xticks(np.arange(8), np.uint16(wavelength[::4]))
    plt.yticks([])
    cb = plt.colorbar()
    cb.ax.set_ylabel('Contrast')
    plt.show()


def data_pre_arbitrary(T_path):
    print("Waiting for data preparation...")
    _, TT_array = load_mat(T_path)
    all_num = TT_array.shape[0]
    all_name_np = TT_array[:, 0]
    all_gap_np = (TT_array[:, 1] - 200) / 200
    all_spec_np = TT_array[:, 2:]
    # all_gauss_np = np.zeros((all_num, 60))
    all_shape_np = np.zeros((all_num, 1, 64, 64))
    all_ctrast_np = np.zeros((all_num, 14))
    with tqdm(total=all_num, ncols=70) as t:
        delete_list = []
        for i in range(all_num):
            # shape
            find = False
            name = str(int(all_name_np[i]))
            filelist = os.listdir('polygon')
            for file in filelist:
                if name == file.split('_')[0]:
                    img_np = cv2.imread('polygon/' + file, cv2.IMREAD_GRAYSCALE)
                    all_shape_np[i, 0, :, :] = img_np / 255
                    find = True
                    break
                else:
                    continue
            if not find:
                print("NO match with " + str(i) + ", it will be deleted later!")
                delete_list.append(i)
            # calculate contrast
            all_ctrast_np[i, :] = np.concatenate((cal_contrast_vector(
                all_spec_np[i, :29]), cal_contrast_vector(all_spec_np[i, 29:])))
            # gauss curve fit
            # try:
            #     all_gauss_np[i, :] = np.concatenate(
            #         (np.array(gauss10_curve_fit(all_spec_np[i, :29])), np.array(gauss10_curve_fit(all_spec_np[i, 29:]))))
            # except:
            #     print("Optimal parameters not found with " + str(i) + ", it will be deleted later!")
            #     if find:
            #         delete_list.append(i)
            t.update()
    # delete error guys
    all_name_np = np.delete(all_name_np, delete_list, axis=0)
    all_gap_np = np.delete(all_gap_np, delete_list, axis=0)
    all_spec_np = np.delete(all_spec_np, delete_list, axis=0)
    all_shape_np = np.delete(all_shape_np, delete_list, axis=0)
    # all_gauss_np = np.delete(all_gauss_np, delete_list, axis=0)
    all_ctrast_np = np.delete(all_ctrast_np, delete_list, axis=0)
    np.save('data/all_gap.npy', all_gap_np)
    np.save('data/all_spec.npy', all_spec_np)
    np.save('data/all_shape.npy', all_shape_np)
    np.save('data/all_ctrast.npy', all_ctrast_np)


def data_enhancement():
    print("Waiting for Data Enhancement...")
    all_gap_org = np.load('data/all_gap.npy')
    all_spec_org = np.load('data/all_spec.npy')
    all_shape_org = np.load('data/all_shape.npy')
    all_spec_90_270 = np.zeros_like(all_spec_org)
    all_shape_90, all_shape_270, all_shape_180 = np.zeros_like(
        all_shape_org), np.zeros_like(all_shape_org), np.zeros_like(all_shape_org)
    for i in range(all_gap_org.shape[0]):
        all_spec_90_270[i, :] = np.concatenate((all_spec_org[i, 29:], all_spec_org[i, :29]), axis=1)
        all_shape_90[i, 0, :, :] = rotate_bound(all_shape_org[i, 0, :, :], 90)
        all_shape_180[i, 0, :, :] = rotate_bound(all_shape_org[i, 0, :, :], 180)
        all_shape_270[i, 0, :, :] = rotate_bound(all_shape_org[i, 0, :, :], 270)
    all_gap_en = np.concatenate((all_gap_org, all_gap_org, all_gap_org, all_gap_org), axis=0)
    all_spec_en = np.concatenate((all_spec_org, all_spec_90_270, all_spec_org, all_spec_90_270), axis=0)
    all_shape_en = np.concatenate((all_shape_org, all_shape_90, all_shape_180, all_shape_270), axis=0)
    np.save('data/all_gap_en.npy', all_gap_en)
    np.save('data/all_spec_en.npy', all_spec_en)
    np.save('data/all_shape_en.npy', all_shape_en)


if __name__ == '__main__':
    all_ctrast = np.load('data/all_ctrast.npy')
    all_spec = np.load('data/all_spec.npy')
    wavelength = np.linspace(400, 680, 29)
    plot_both_parts_2(wavelength, all_spec[0, :29], all_ctrast[0, :7], 'contrast_vector.png')
