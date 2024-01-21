# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:08:19 2020

@author: 77509
"""
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu,sigmoid
import torch.nn.functional as F
from myfilter import myfilter


class ica_erdsNet(nn.Module):
    def __init__(self,
                 in_chans,
                 n_classes,
                 input_time_length,
                 crop_length,

                 n_filters_ica=64,
                 filters_ica_stride=1,
                 filters=[],
                 filters_time_size=11,

                 poolmax_size=80,
                 poolmax_stride=16,

                 batch_norm=True,
                 batch_norm_alpha=0.1,

                 drop_prob1=0.5, ):
        super(ica_erdsNet, self).__init__()
        self.__dict__.update(locals())
        del self.self

        self.n_filters_time = len(filters)

        # 独立成分分析
        self.conv_ica = nn.Conv2d(1, self.n_filters_ica, (1, self.in_chans), stride=(self.filters_ica_stride, 1),
                                  bias=False, )

        # 滤波，分波段ERDS
        self.conv_time = nn.Conv2d(1, self.n_filters_time, (self.filters_time_size, 1), stride=(1, 1), bias=False, )
        # 批归一化、激活、池化、失活
        self.batch1 = nn.BatchNorm2d(self.n_filters_ica, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, )
        self.batch2 = nn.BatchNorm2d(self.n_filters_time, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, )
        # 激活
        new_crop_length = int(self.crop_length - self.filters_time_size + 1)

        # self.poolmax = nn.MaxPool2d(kernel_size=(new_crop_length, 1), stride=(self.poolmax_stride,  1))
        self.poolmean = nn.AvgPool2d(kernel_size=(new_crop_length, 1), stride=(self.poolmax_stride, 1))
        self.poolmean2 = nn.AvgPool2d(kernel_size=(int(self.input_time_length - self.filters_time_size + 1), 1),
                                      stride=(self.poolmax_stride, 1))
        self.poolmean_ica = nn.AvgPool2d(kernel_size=(80, 1), stride=(16, 1))

        self.dropout1 = nn.Dropout(drop_prob1)

        # 分类卷积层
        self.conv_class = nn.Conv2d(self.n_filters_time,
                                    self.n_classes,
                                    kernel_size=(1, self.n_filters_ica),
                                    stride=(1, 1),
                                    bias=False, )

        # print(self.first_pool_stride)
        self.softmax = nn.LogSoftmax(dim=1)

        # nn.init.constant_(self.conv_ica.bias, 0)
        nn.init.xavier_uniform_(self.conv_ica.weight, gain=1)
        # nn.init.constant_(self.conv_time.bias, 0)
        nn.init.xavier_uniform_(self.conv_time.weight, gain=1)

        # 初始化滤波
        fliter_n=self.filters_time_size
        fil = np.zeros([self.n_filters_time, 1, fliter_n, 1])
        for i,lh in enumerate(self.filters):
            fil1= myfilter(fs=160, fl=lh[0], fh=lh[1], fliter_n=fliter_n, mode ='N')
            fil[i,0,:,:] = fil1.reshape([fliter_n,1])
        time_tensor = torch.Tensor(fil)
        self.conv_time.weight = torch.nn.Parameter(time_tensor)

        '''time_weight = np.load('w_fil.npy')
        time_weight =  time_weight.reshape([-1,1,31,1])
        self.conv_time.weight = torch.nn.Parameter(torch.tensor(time_weight))'''
        # 初始化ica  16,1,1,64
        '''ica_weight = np.load('initial_ica.npy')
        self.conv_ica.weight = torch.nn.Parameter(torch.tensor(ica_weight))'''

        '''ww = mixing_matrix(0.2, -0.4, al = 0.8)
        ica_weight = torch.tensor(ww[[1,3,5,8,10,12],:64].reshape([6,1,1,64]), dtype=torch.float)
        self.conv_ica.weight = torch.nn.Parameter(ica_weight)'''

        nn.init.constant_(self.batch1.weight, 1)
        nn.init.constant_(self.batch1.bias, 0)

        # self.conv_ica.weight.data[:,:,:,21:28] = torch.nn.Parameter(-torch.abs(self.conv_ica.weight.data[:,:,:,21:28]))

        nn.init.xavier_uniform_(self.conv_class.weight, gain=1)
        # nn.init.constant_(self.conv_class.bias, 0)

        '''ica_weight = np.load('w_class.npy')
        self.conv_class.weight = torch.nn.Parameter(torch.tensor(ica_weight))'''

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)

        x = self.conv_ica(x)
        x = x.permute(0, 3, 2, 1)

        '''xe = self.batch2(x)
        energy = self.poolmean_ica(torch.mul(xe,xe))
        emax,n = torch.max(energy,2)
        emin,n = torch.min(energy,2)
        e = torch.mean((emax-emin))'''

        x = self.conv_time(x)

        '''限幅'''
        # print(x.data.size())
        '''std = torch.mean(torch.mul(x,x), 2)
        std = std.sqrt()
        std = std*(2+self.n/40)
        std = std.unsqueeze(2)
        std = std.repeat(1,1,610,1)
        x = torch.div(x,std)*2
        x = torch.tanh(x)
        x = torch.mul(x,std)'''
        # print(std.data.size())
        x = x.permute(0, 3, 2, 1)
        x = self.batch1(x)
        x = x.permute(0, 3, 2, 1)
        x = torch.mul(x, x)

        y = self.poolmean2(x)
        y = self.softmax(self.conv_class(y))
        y = y.squeeze()

        '''x = torch.log(torch.clamp(x, min=1e-6))
        x = self.batch2(x)'''

        x = self.poolmean(x)
        '''energy = torch.mean(x)'''
        # x = torch.cat((x1,x2), 1)

        # x = self.dropout1(x)
        # a = random.sample(range(int(self.n_filters_spat)),int(self.n_filters_spat//1))
        # x = self.dropout1(x)
        x = self.softmax(self.conv_class(x))
        x = x.squeeze()

        '''w = self.conv_class.weight.data
        w = self.dropout1(w)*0.1
        self.conv_class.weight = torch.nn.Parameter(w)'''

        return x, y, 0


class ica_erdsNet1(nn.Module):
    def __init__(self,
                 in_chans,
                 n_classes,
                 input_time_length,
                 crop_length,

                 n_filters_ica=64,
                 filters_ica_stride=1,
                 filters=[],
                 filters_time_size=11,

                 poolmax_size=80,
                 poolmax_stride=16,

                 batch_norm=True,
                 batch_norm_alpha=0.1,

                 drop_prob1=0.5, ):
        super(ica_erdsNet1, self).__init__()
        self.__dict__.update(locals())
        del self.self

        self.n_filters_time = len(filters)

        # 独立成分分析
        self.conv_ica = nn.Conv2d(1, self.n_filters_ica, (1, self.in_chans), stride=(self.filters_ica_stride, 1),
                                  bias=False, )

        # 滤波，分波段ERDS
        self.conv_time = nn.Conv2d(1, self.n_filters_time, (self.filters_time_size, 1), stride=(1, 1), bias=False, )
        self.conv_time.weight.requires_grad = False
        # 批归一化、激活、池化、失活
        self.batch1 = nn.BatchNorm2d(self.n_filters_time, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, )
        self.batch2 = nn.BatchNorm2d(self.n_filters_time, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, )
        # 激活
        new_crop_length = int(self.crop_length - self.filters_time_size + 1)

        # self.poolmax = nn.MaxPool2d(kernel_size=(new_crop_length, 1), stride=(self.poolmax_stride,  1))
        self.poolmean = nn.AvgPool2d(kernel_size=(new_crop_length, 1), stride=(self.poolmax_stride, 1))
        self.poolmean2 = nn.AvgPool2d(kernel_size=(int(self.input_time_length - self.filters_time_size + 1), 1),
                                      stride=(self.poolmax_stride, 1))
        self.poolmean_ica = nn.AvgPool2d(kernel_size=(80, 1), stride=(16, 1))

        self.dropout1 = nn.Dropout(drop_prob1)

        # 分类卷积层
        self.conv_class = nn.Conv2d(self.n_filters_time,
                                    self.n_classes,
                                    kernel_size=(1, self.n_filters_ica),
                                    stride=(1, 1),
                                    bias=False, )

        # print(self.first_pool_stride)
        self.softmax = nn.LogSoftmax(dim=1)

        # nn.init.constant_(self.conv_ica.bias, 0)
        nn.init.xavier_uniform_(self.conv_ica.weight, gain=1)
        # nn.init.constant_(self.conv_time.bias, 0)
        nn.init.xavier_uniform_(self.conv_time.weight, gain=1)

        # 初始化滤波
        fliter_n=self.filters_time_size
        fil = np.zeros([self.n_filters_time, 1, fliter_n, 1])
        for i,lh in enumerate(self.filters):
            fil1= myfilter(fs=160, fl=lh[0], fh=lh[1], fliter_n=fliter_n, mode ='N')
            fil[i,0,:,:] = fil1.reshape([fliter_n,1])
        time_tensor = torch.Tensor(fil)
        self.conv_time.weight = torch.nn.Parameter(time_tensor)

        '''time_weight = np.load('w_fil.npy')
        time_weight =  time_weight.reshape([-1,1,31,1])
        self.conv_time.weight = torch.nn.Parameter(torch.tensor(time_weight))'''
        # 初始化ica  16,1,1,64
        '''ica_weight = np.load('initial_ica.npy')
        self.conv_ica.weight = torch.nn.Parameter(torch.tensor(ica_weight))'''

        '''ww = mixing_matrix(0.2, -0.4, al = 0.8)
        ica_weight = torch.tensor(ww[[1,3,5,8,10,12],:64].reshape([6,1,1,64]), dtype=torch.float)
        self.conv_ica.weight = torch.nn.Parameter(ica_weight)'''

        nn.init.constant_(self.batch1.weight, 1)
        nn.init.constant_(self.batch1.bias, 0)

        # self.conv_ica.weight.data[:,:,:,21:28] = torch.nn.Parameter(-torch.abs(self.conv_ica.weight.data[:,:,:,21:28]))

        nn.init.xavier_uniform_(self.conv_class.weight, gain=1)
        # nn.init.constant_(self.conv_class.bias, 0)

        '''ica_weight = np.load('w_class.npy')
        self.conv_class.weight = torch.nn.Parameter(torch.tensor(ica_weight))'''

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)

        x = self.conv_ica(x)
        x = x.permute(0, 3, 2, 1)

        x = self.conv_time(x)

        x = self.batch1(x)
        x = torch.mul(x, x)

        x = self.poolmean(x)

        x = self.softmax(self.conv_class(x))
        x = x.squeeze()

        return x, 0, 0

'''attention'''
class ica_erdsNet2(nn.Module):
    def __init__(self,
                 in_chans,
                 n_classes,
                 input_time_length,
                 crop_length,
        
                 n_filters_ica = 64,
                 filters_ica_stride = 1,
                 filters = [],
                 filters_time_size = 11,
                
                 poolmax_size = 80,
                 poolmax_stride = 16,
        
                 batch_norm=True,
                 batch_norm_alpha=0.1, 
        
                 drop_prob1=0.5,):
        
        super(ica_erdsNet2, self).__init__()
        self.__dict__.update(locals())
        del self.self
        
        self.n_filters_time = len(filters)

        # 独立成分分析
        self.conv_ica = nn.Conv2d( 1, self.n_filters_ica, (1, self.in_chans), stride=(self.filters_ica_stride,1),bias=False,)
        
        # 滤波，分波段ERDS
        self.conv_time = nn.Conv2d(1, self.n_filters_time, (self.filters_time_size, 1), stride=(1, 1), bias=False, )
        self.conv_time.weight.requires_grad = False
        # 批归一化、激活、池化、失活
        self.batch1 = nn.BatchNorm2d(self.n_filters_time, momentum=self.batch_norm_alpha,affine=True,eps=1e-5,)
        self.batch2 = nn.BatchNorm2d(self.n_filters_time, momentum=self.batch_norm_alpha,affine=True,eps=1e-5,)
        # 激活
        new_crop_length = int(self.crop_length - self.filters_time_size+1)
        
        #self.poolmax = nn.MaxPool2d(kernel_size=(new_crop_length, 1), stride=(self.poolmax_stride,  1))
        self.poolmean = nn.AvgPool2d(kernel_size=(new_crop_length, 1), stride=(self.poolmax_stride, 1))
        self.poolmean2 = nn.AvgPool2d(kernel_size=(int(self.input_time_length - self.filters_time_size+1), 1), stride=(self.poolmax_stride, 1))
        self.poolmean_ica = nn.AvgPool2d(kernel_size=(80, 1), stride=(16, 1))
        
        self.dropout1 = nn.Dropout(drop_prob1)
        
        
        # 分类卷积层
        self.conv_class = nn.Conv2d(  self.n_filters_time,
                                      self.n_classes,
                                      kernel_size = (1, self.n_filters_ica),
                                      stride = (1,1),
                                      bias=False,)
    
        #print(self.first_pool_stride)
        self.softmax = nn.LogSoftmax(dim=1)
        
        
        #nn.init.constant_(self.conv_ica.bias, 0)
        nn.init.xavier_uniform_(self.conv_ica.weight, gain=1)
        #nn.init.constant_(self.conv_time.bias, 0)
        nn.init.xavier_uniform_(self.conv_time.weight, gain=1)

        # 初始化滤波  
        fliter_n=self.filters_time_size
        fil = np.zeros([self.n_filters_time, 1, fliter_n, 1])
        for i,lh in enumerate(self.filters):
            fil1= myfilter(fs=160, fl=lh[0], fh=lh[1], fliter_n=fliter_n, mode ='N')
            fil[i,0,:,:] = fil1.reshape([fliter_n,1])
        time_tensor = torch.Tensor(fil)
        self.conv_time.weight = torch.nn.Parameter(time_tensor)
        
        '''time_weight = np.load('w_fil.npy')
        time_weight =  time_weight.reshape([-1,1,31,1])
        self.conv_time.weight = torch.nn.Parameter(torch.tensor(time_weight))'''
        # 初始化ica  16,1,1,64
        '''ica_weight = np.load('initial_ica.npy')
        self.conv_ica.weight = torch.nn.Parameter(torch.tensor(ica_weight))'''
        
        '''ww = mixing_matrix(0.2, -0.4, al = 0.8)
        ica_weight = torch.tensor(ww[[1,3,5,8,10,12],:64].reshape([6,1,1,64]), dtype=torch.float)
        self.conv_ica.weight = torch.nn.Parameter(ica_weight)'''
        
        
        nn.init.constant_(self.batch1.weight, 1)
        nn.init.constant_(self.batch1.bias, 0)
        
        #self.conv_ica.weight.data[:,:,:,21:28] = torch.nn.Parameter(-torch.abs(self.conv_ica.weight.data[:,:,:,21:28]))
        
        
        nn.init.xavier_uniform_(self.conv_class.weight, gain=1)
        #nn.init.constant_(self.conv_class.bias, 0)
        
        '''ica_weight = np.load('w_class.npy')
        self.conv_class.weight = torch.nn.Parameter(torch.tensor(ica_weight))'''
        
    def forward(self, x):
        x = x.permute(0, 3, 2, 1)

        x = self.conv_ica(x)
        x = x.permute(0, 3, 2, 1)
        
        '''xe = self.batch2(x)
        energy = self.poolmean_ica(torch.mul(xe,xe))
        emax,n = torch.max(energy,2)
        emin,n = torch.min(energy,2)
        e = torch.mean((emax-emin))'''
        
        x = self.conv_time(x)
        
        '''限幅'''
        #print(x.data.size())
        '''std = torch.mean(torch.mul(x,x), 2)
        std = std.sqrt()
        std = std*(2+self.n/40)
        std = std.unsqueeze(2)
        std = std.repeat(1,1,610,1)
        x = torch.div(x,std)*2
        x = torch.tanh(x)
        x = torch.mul(x,std)'''
        #print(std.data.size())
        
        
        x = self.batch1(x)
        x = torch.mul(x,x)
        
        y = self.poolmean2(x)
        y = self.softmax(self.conv_class(y))
        y = y.squeeze()
        
        '''x = torch.log(torch.clamp(x, min=1e-6))
        x = self.batch2(x)'''
        #print(x.data.size())
        x = self.poolmean(x)
        '''energy = torch.mean(x)'''
        #x = torch.cat((x1,x2), 1)
        #x = self.dropout1(x)
        #a = random.sample(range(int(self.n_filters_spat)),int(self.n_filters_spat//1))
        #x = self.dropout1(x)
        x = self.softmax(self.conv_class(x))
        x = x.squeeze()
        
        '''w = self.conv_class.weight.data
        w = self.dropout1(w)*0.1
        self.conv_class.weight = torch.nn.Parameter(w)'''
        
        return x, y, 0


class ica_erdsNet_show(nn.Module):
    def __init__(self,
                 in_chans,
                 n_classes,
                 input_time_length,
                 crop_length,
        
                 n_filters_ica = 64,
                 filters_ica_stride = 1,
                 filters = [],
                 filters_time_size = 11,
                
                 poolmax_size = 80,
                 poolmax_stride = 16,
        
                 batch_norm=True,
                 batch_norm_alpha=0.1, 
        
                 drop_prob1=0.5,):
        
        super(ica_erdsNet_show, self).__init__()
        self.__dict__.update(locals())
        del self.self
        
        self.n_filters_time = len(filters)

        # 独立成分分析
        self.conv_ica = nn.Conv2d( 1, self.n_filters_ica, (1, self.in_chans), stride=(self.filters_ica_stride,1),bias=False,)
        
        # 滤波，分波段ERDS
        self.conv_time = nn.Conv2d(1, self.n_filters_time, (self.filters_time_size, 1), stride=(1, 1), bias=False, )
        self.conv_time.weight.requires_grad = False
        # 批归一化、激活、池化、失活
        self.batch1 = nn.BatchNorm2d(self.n_filters_time, momentum=self.batch_norm_alpha,affine=True,eps=1e-5,)
        self.batch2 = nn.BatchNorm2d(self.n_filters_time, momentum=self.batch_norm_alpha,affine=True,eps=1e-5,)
        # 激活
        new_crop_length = int(self.crop_length - self.filters_time_size+1)
        
        #self.poolmax = nn.MaxPool2d(kernel_size=(new_crop_length, 1), stride=(self.poolmax_stride,  1))
        self.poolmean = nn.AvgPool2d(kernel_size=(64, 1), stride=(self.poolmax_stride, 1))
        self.poolmean2 = nn.AvgPool2d(kernel_size=(160, 1), stride=(160, 1))
        self.poolmean_ica = nn.AvgPool2d(kernel_size=(80, 1), stride=(16, 1))
        
        self.dropout1 = nn.Dropout(drop_prob1)
        
        
        # 分类卷积层
        self.conv_class = nn.Conv2d(  self.n_filters_time,
                                      self.n_classes,
                                      kernel_size = (1, self.n_filters_ica),
                                      stride = (1,1),
                                      bias=False,)
    
        #print(self.first_pool_stride)
        self.softmax = nn.LogSoftmax(dim=1)
        
        mm = torch.load('model.pth')
        
        #nn.init.constant_(self.conv_ica.bias, 0)
        nn.init.xavier_uniform_(self.conv_ica.weight, gain=1)
        #nn.init.constant_(self.conv_time.bias, 0)
        nn.init.xavier_uniform_(self.conv_time.weight, gain=1)


        self.conv_time.weight = mm.conv_time.weight
        w_ica = mm.conv_ica.weight.data.numpy()
        '''informax = np.load('w_informax.npy')
        w_ica [6,0,0,:] = informax[7,:]'''
        self.conv_ica.weight= torch.nn.Parameter(torch.tensor(w_ica))
        
        self.conv_class.weight = mm.conv_class.weight
        self.batch1.weight = mm.batch1.weight
        self.batch1.bias = mm.batch1.bias
        

        
        '''ica_weight = np.load('w_class.npy')
        self.conv_class.weight = torch.nn.Parameter(torch.tensor(ica_weight))'''
        
    def forward(self, x):
        x = x.permute(0, 3, 2, 1)

        x = self.conv_ica(x)
        x = x.permute(0, 3, 2, 1)
        
        x = self.conv_time(x)
        
        x = self.batch1(x)
        x = torch.mul(x,x)
        
        y = self.poolmean2(x)
        
        #x = x - y[:,:,0:1,:]
        x = torch.log(self.poolmean(x))-torch.log(y[:,:,0:1,:])
        #x = torch.div(x, y[:,:,0:1,:])
        #x = x[:,:,-35:,:]

        z = self.softmax(self.conv_class(x))
        z = z.squeeze()
    
        
        return x, z


class ica_erdsNet_se(nn.Module):
    def __init__(self,
                 in_chans,
                 n_classes,
                 input_time_length,
                 crop_length,

                 n_filters_ica=64,
                 filters_ica_stride=1,
                 filters=[],
                 filters_time_size=11,

                 poolmax_size=80,
                 poolmax_stride=16,

                 batch_norm=True,
                 batch_norm_alpha=0.1,

                 drop_prob1=0.5, ):
        super(ica_erdsNet_se, self).__init__()
        self.__dict__.update(locals())
        del self.self

        self.n_filters_time = len(filters)

        # 独立成分分析
        self.conv_ica = nn.Conv2d(1, self.n_filters_ica, (1, self.in_chans), stride=(self.filters_ica_stride, 1),
                                  bias=False, )
        self.conv_ica.weight.requires_grad = False
        # 滤波，分波段ERDS
        self.conv_time = nn.Conv2d(1, self.n_filters_time, (self.filters_time_size, 1), stride=(1, 1), bias=False, )
        self.conv_time.weight.requires_grad = False
        # 批归一化、激活、池化、失活
        self.batch1 = nn.BatchNorm2d(self.n_filters_time, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, )
        self.batch2 = nn.BatchNorm2d(self.n_filters_time, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, )
        # 激活
        new_crop_length = int(self.crop_length - self.filters_time_size + 1)

        # self.poolmax = nn.MaxPool2d(kernel_size=(new_crop_length, 1), stride=(self.poolmax_stride,  1))
        self.poolmean = nn.AvgPool2d(kernel_size=(new_crop_length, 1), stride=(self.poolmax_stride, 1))
        self.poolmean2 = nn.AvgPool2d(kernel_size=(int(self.input_time_length - self.filters_time_size + 1), 1),
                                      stride=(self.poolmax_stride, 1))
        self.poolmean_ica = nn.AvgPool2d(kernel_size=(80, 1), stride=(16, 1))

        self.dropout1 = nn.Dropout(drop_prob1)

        # 分类卷积层
        self.conv_class = nn.Conv2d(self.n_filters_time,
                                    self.n_classes,
                                    kernel_size=(1, self.n_filters_ica),
                                    stride=(1, 1),
                                    bias=False, )

        # print(self.first_pool_stride)
        self.softmax = nn.LogSoftmax(dim=1)

        self.se1 = nn.Conv2d(self.n_filters_time, self.n_filters_time, (1, self.n_filters_ica),
                             groups=self.n_filters_time, )
        self.se2 = nn.Conv2d(self.n_filters_time, self.n_filters_time, (1, 1), )

        nn.init.constant_(self.se1.bias, 0)
        nn.init.xavier_uniform_(self.se1.weight, gain=1)
        nn.init.constant_(self.se2.bias, 0)
        nn.init.xavier_uniform_(self.se2.weight, gain=1)

        # 初始化滤波
        fliter_n = self.filters_time_size
        fil = np.zeros([self.n_filters_time, 1, fliter_n, 1])
        for i, lh in enumerate(self.filters):
            fil1 = myfilter(fs=160, fl=lh[0], fh=lh[1], fliter_n=fliter_n, mode='N')
            fil[i, 0, :, :] = fil1.reshape([fliter_n, 1])
        time_tensor = torch.Tensor(fil)
        self.conv_time.weight = torch.nn.Parameter(time_tensor)

        '''time_weight = np.load('w_fil.npy')
        time_weight =  time_weight.reshape([-1,1,31,1])
        self.conv_time.weight = torch.nn.Parameter(torch.tensor(time_weight))'''
        # 初始化ica  16,1,1,64
        '''complate = torch.load('model.pth')
        self.conv_ica.weight = complate.conv_ica.weight'''
        '''ica_weight = np.load('w_ica.npy')
        self.conv_ica.weight = torch.nn.Parameter(torch.tensor(ica_weight))'''

        '''ww = mixing_matrix(0.2, -0.4, al = 0.8)
        ica_weight = torch.tensor(ww[[1,3,5,8,10,12],:64].reshape([6,1,1,64]), dtype=torch.float)
        self.conv_ica.weight = torch.nn.Parameter(ica_weight)'''

        '''self.batch1.weight = complate.batch1.weight
        self.batch1.bias = complate.batch1.bias'''

        # self.conv_ica.weight.data[:,:,:,21:28] = torch.nn.Parameter(-torch.abs(self.conv_ica.weight.data[:,:,:,21:28]))

        #nn.init.xavier_uniform_(self.conv_class.weight, gain=1)
        # nn.init.constant_(self.conv_class.bias, 0)

        '''ica_weight = np.load('w_class.npy')
        self.conv_class.weight = torch.nn.Parameter(torch.tensor(ica_weight))'''
        #self.conv_class.weight = complate.conv_class.weight

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)

        x = self.conv_ica(x)
        x = x.permute(0, 3, 2, 1)

        x = self.conv_time(x)
        
        x = self.batch1(x)
        x = torch.mul(x, x)

        y = self.poolmean2(x)
        y = torch.sigmoid(self.se1(y))
        y = F.elu(self.se2(y))

        x = self.poolmean(x)

        x = x * y.expand_as(x)
        x = self.softmax(self.conv_class(x))
        x = x.squeeze()

        return x, y, 0