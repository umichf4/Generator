# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:44:24 2019

@author: Pi
"""
import torch
import torch.nn as nn

class G(nn.Module):

    def __init__(self, in_features=256, out_features=5):
        super(G, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.FC1 = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
        )
        self.Conv = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=4, stride=2,
                 padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=4, stride=2,
                 padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=4, kernel_size=4, stride=2,
                 padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(16, self.out_features)
        )
        weights_init(self)

    def forward(self, spec, noise):
        x = torch.cat([spec, noise], 1)
        x = self.FC1(x)
        x = x.view(-1, 64, 32)
        x = self.Conv(x)
        x = x.view(-1, 16)
        x = self.FC2(x)
        x = torch.tanh(x)
        return x

def weights_init(m):  # Initiate weights
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
    
    generator = G()
        
    shape = torch.ones([4, 2])
    spec = torch.ones([4, 57])
    z = torch.ones([4, 199])
    
    out = generator(shape, spec, z)
    print(generator)