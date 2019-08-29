# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:02:20 2019

@author: Pi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class conv_bn_relu(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride, padding, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv1d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)

        return out
    
class Transition_layer(nn.Sequential):
  def __init__(self, nin, theta=1):    
      super(Transition_layer, self).__init__()
      
      self.add_module('conv_1x1', conv_bn_relu(nin=nin, nout=int(nin*theta), kernel_size=1, stride=1, padding=0, bias=False))
      self.add_module('avg_pool_2x2', nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
      
class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        
        self.conv_3x3_first = conv_bn_relu(nin=1, nout=32, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.conv_1x1_left = conv_bn_relu(nin=32, nout=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3x3_left = conv_bn_relu(nin=16, nout=32, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.max_pool_right = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.conv_1x1_last = conv_bn_relu(nin=64, nout=32, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out_first = self.conv_3x3_first(x)
        
        out_left = self.conv_1x1_left(out_first)
        out_left = self.conv_3x3_left(out_left)
        
        out_right = self.max_pool_right(out_first)
        
        out_middle = torch.cat((out_left, out_right), 1)
        
        out_last = self.conv_1x1_last(out_middle)
                
        return out_last
    
class dense_layer(nn.Module):
  def __init__(self, nin, growth_rate, drop_rate=0.2):    
      super(dense_layer, self).__init__()
      
      self.dense_left_way = nn.Sequential()
      
      self.dense_left_way.add_module('conv_1x1', conv_bn_relu(nin=nin, nout=growth_rate*2, kernel_size=1, stride=1, padding=0, bias=False))
      self.dense_left_way.add_module('conv_3x3', conv_bn_relu(nin=growth_rate*2, nout=growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False))
            
      self.dense_right_way = nn.Sequential()
      
      self.dense_right_way.add_module('conv_1x1', conv_bn_relu(nin=nin, nout=growth_rate*2, kernel_size=1, stride=1, padding=0, bias=False))
      self.dense_right_way.add_module('conv_3x3_1', conv_bn_relu(nin=growth_rate*2, nout=growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False))
      self.dense_right_way.add_module('conv_3x3 2', conv_bn_relu(nin=growth_rate//2, nout=growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False))
      
      self.drop_rate = drop_rate
      
  def forward(self, x):
      left_output = self.dense_left_way(x)
      right_output = self.dense_right_way(x)

      if self.drop_rate > 0:
          left_output = F.dropout(left_output, p=self.drop_rate, training=self.training)
          right_output = F.dropout(right_output, p=self.drop_rate, training=self.training)
          
      dense_layer_output = torch.cat((x, left_output, right_output), 1)
            
      return dense_layer_output
  
class DenseBlock(nn.Sequential):
  def __init__(self, nin, num_dense_layers, growth_rate, drop_rate=0.0):
      super(DenseBlock, self).__init__()
                        
      for i in range(num_dense_layers):
          nin_dense_layer = nin + growth_rate * i
          self.add_module('dense_layer_%d' % i, dense_layer(nin=nin_dense_layer, growth_rate=growth_rate, drop_rate=drop_rate))
          
class PeleeNet(nn.Module):
    def __init__(self, growth_rate=32, num_dense_layers=[3,4,8,6], theta=1, drop_rate=0.0, num_classes=10):
        super(PeleeNet, self).__init__()
        
        assert len(num_dense_layers) == 4
        
        self.features = nn.Sequential()
        self.features.add_module('StemBlock', StemBlock())
        
        nin_transition_layer = 32
        
        for i in range(len(num_dense_layers)):
            self.features.add_module('DenseBlock_%d' % (i+1), DenseBlock(nin=nin_transition_layer, num_dense_layers=num_dense_layers[i], growth_rate=growth_rate, drop_rate=0.0))
            nin_transition_layer +=  num_dense_layers[i] * growth_rate
            
            if i == len(num_dense_layers) - 1:
                self.features.add_module('Transition_layer_%d' % (i+1), conv_bn_relu(nin=nin_transition_layer, nout=int(nin_transition_layer*theta), kernel_size=1, stride=1, padding=0, bias=False))
            else:
                self.features.add_module('Transition_layer_%d' % (i+1), Transition_layer(nin=nin_transition_layer, theta=1))
        
        self.linear = nn.Linear(nin_transition_layer, num_classes)
        
    def forward(self, x):
        stage_output = self.features(x)
        
        global_avg_pool_output = F.adaptive_avg_pool1d(stage_output, 1)  
        global_avg_pool_output_flat = global_avg_pool_output.view(global_avg_pool_output.size(0), -1)
                
        output = self.linear(global_avg_pool_output_flat)
        
        return output

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = PeleeNet(num_classes=2)
    net.to(device)
    torchsummary.summary(net, (1, 224))