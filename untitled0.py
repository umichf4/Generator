# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 13:23:08 2019

@author: Pi
"""
from utils import *
import random
import matplotlib.pyplot as plt
import numpy as np

spec_dim = 56
spec_x = np.linspace(0, 1, 56)
noise_dim = 200

mean = random.uniform(0, 0.5)
sigma = random.uniform(0, 0.5)
spec = normal_distribution(spec_x, mean, sigma)
plt.plot(spec_x, spec, label='1')
plt.legend()
plt.grid()
plt.show()