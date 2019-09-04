# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:44:24 2019

@author: Pi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(model):
    for m in model.modules():
        if isinstance(m,nn.Conv1d):
            #nn.init.normal(m.weight.data)
            #nn.init.xavier_normal(m.weight.data)
            #nn.init.kaiming_normal(m.weight.data)
            m.weight.data.normal_()
            m.bias.data.fill_(0)
            
        elif isinstance(m,nn.Linear):
            m.weight.data.normal_()
            
        elif isinstance(m,nn.BatchNorm1d):
            m.weight.data.normal_()
            m.bias.data.fill_(0)
            
        elif isinstance(m,nn.ConvTranspose1d):
            m.weight.data.normal_()
            m.bias.data.fill_(0)
            
class G_FC_Conv_Fc(nn.Module):

    def __init__(self, in_features=256, out_features=5):
        super(G_FC_Conv_Fc, self).__init__()
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
        
        self.FC3 = nn.Sequential(
            nn.Linear(16, self.out_features)
        )
        
        weights_init(self)

    def forward(self, spec, noise):
        x = torch.cat([spec, noise], 1)
        x = self.FC1(x)
        x = x.view(-1, 64, 32)
        x = self.Conv(x)
        x = x.view(-1, 16)
        mu = self.FC2(x)
        mu = torch.tanh(mu)
        var = self.FC3(x)
        var = F.softplus(var)
        return mu, var

class G_Deconv_Fc(nn.Module):

    def __init__(self, in_features=128, out_features=5):
        super(G_Deconv_Fc, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.Deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=4, 
                               stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=4, 
                               stride=2, padding=1),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=4, 
                               stride=2, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )
            
        self.FC = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Linear(256, self.out_features),
        )
        
        self.FC_mu = nn.Sequential(
            nn.Linear(256, self.out_features)    
        )
        
        self.FC_var = nn.Sequential(
            nn.Linear(256, self.out_features)    
        )

        weights_init(self)

    def forward(self, spec, noise):
        x = torch.cat([spec, noise], 1)
        assert x.shape[-1] == 128
        x = x.view(-1, 8, 16)
        x = self.Deconv(x)
        x = x.squeeze(1)
        x = self.FC(x)
        mu = self.FC_mu(x)
        mu = torch.sigmoid(mu)
        var = self.FC_var(x)
        var = torch.sigmoid(var)
        return mu, var
    

if __name__ == '__main__':
    
    generator = G_Deconv_Fc()
        
    shape = torch.ones([4, 2])
    spec = torch.ones([4, 28])
    z = torch.ones([4, 100])
    
    out = generator(spec, z)
    print(generator)
#    generator = G_FC_Conv_Fc()
#        
#    shape = torch.ones([4, 2])
#    spec = torch.ones([4, 56])
#    z = torch.ones([4, 200])
#    
#    out = generator(spec, z)
#    print(generator)