# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-30 16:55:57
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-09-06 23:59:21

import numpy as np
from utils import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2


def gauss10(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9):

    return a0 * np.exp(-((x - m0) / s0)**2) + a1 * np.exp(-((x - m1) / s1)**2) + a2 * np.exp(-((x - m2) / s2)**2) + \
        a3 * np.exp(-((x - m3) / s3)**2) + a4 * np.exp(-((x - m4) / s4)**2) + a5 * np.exp(-((x - m5) / s5)**2) + \
        a6 * np.exp(-((x - m6) / s6)**2) + a7 * np.exp(-((x - m7) / s7)**2) + a8 * np.exp(-((x - m8) / s8)**2) + \
        a9 * np.exp(-((x - m9) / s9)**2)


_, TT_array = load_mat("data\\shape_spec_3211.mat")
wavelength = np.linspace(400, 680, 29)

all_num = TT_array.shape[0]

x = TT_array[:, :2]
TE_spec = TT_array[:, 2:31]
TM_spec = TT_array[:, 31:]

# for i in range(all_num):
#     plot_both_parts(wavelength, TE_spec[i, :], TM_spec[i, :], str(x[i, 0]) + '_' + str(x[i, 1]) + '.png')
#     print(i)

a_min = [0] * 10
a_max = [1] * 10
m_min = [400] * 10
m_max = [680] * 10
s_min = [0] * 10
s_max = [1000] * 10
min_limit = a_min + m_min + s_min
max_limit = a_max + m_max + s_max

TE_real = TE_spec[16, :]
popt, pcov = curve_fit(gauss10, wavelength, TE_real, bounds=(min_limit, max_limit))
print(popt)
TE_fake = gauss10(wavelength, *popt)
plot_both_parts(wavelength, TE_real, TE_fake, "hhh.png")

# avg = (TE_spec + TM_spec) / 2
# min_index = np.argmin(TM_spec, axis=1)
# min_sort = np.argsort(min_index)
# TE_spec = TM_spec[min_sort, :]

# # TE_spec = cv2.resize(src=TE_spec, dsize=(1000, 1881), interpolation=cv2.INTER_CUBIC)

# plt.figure()
# plt.pcolor(TE_spec, cmap=plt.cm.jet)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Index of Devices')
# plt.title('Possible Spectrums of arbitrary shapes (TM mode)')
# plt.xticks(np.arange(len(wavelength), step=4), np.uint16(wavelength[::4]))
# # plt.yticks([])
# cb = plt.colorbar()
# cb.ax.set_ylabel('Transmittance')
# plt.show()
