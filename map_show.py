# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:06:04 2020

@author: 77509
"""
from pre_trail import create_trial,create_trial2
#from pre_trail_noeog import create_trial
from trialNet import ica_erdsNet_show,ica_erdsNet2
from my_dataset import data_nopath
import matplotlib.pyplot as plt

import numpy as np
from myfilter import myfilter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy
from scipy.fftpack import fft
import time
import os
import seaborn as sns



def FFT (Fs,data):
    L = len (data)                          # 信号长度
    N =np.power(2,np.ceil(np.log2(L)))*4      # 下一个最近二次幂
    N = N.astype('int')
    FFT_y = (fft(data,N))/L*2                  # N点FFT，但除以实际信号长度 L
    Fre = np.arange(int(N/2))*Fs/N          # 频率坐标
    FFT_y = FFT_y[range(int(N/2))]          # 取一半
    
    mode = abs(FFT_y)
    ang = np.angle(FFT_y)
    return Fre, mode,ang

crop_l = 96
trail_l = 640+190
filters_time_size = 131
clas = 5
ff = []
for i in range(4,64,1):
    ff.append([i//2-0.5,1//2+0.5])
filter_list = ff
batch_size1 = 10
batch_size2 = 14

train_x,train_y,test_x,test_y = create_trial2(sub_list=[1], m_t=0, length = 640, n_test=1, kf=1, clas=clas, m=0)

'''_,_,ss = scipy.signal.stft(test_x[2,8,-640:],fs=160,window='hann',nperseg=96,noverlap=80)
sns.heatmap(abs(ss[:20,:]))'''


test = data_nopath(test_x,test_y, batch_size2, False)

'''model = ica_erdsNet_show(64, clas, input_time_length=trail_l, crop_length = crop_l,
                                n_filters_ica = 8,
                                filters = filter_list,
                                filters_time_size = filters_time_size ,
                                poolmax_size=80,poolmax_stride=16,
                                drop_prob1=0.5)'''

model = torch.load('modelS1-2.pth')
crop_n = 35
correct = 0
total = 0
crop_c = 0
crop_t = batch_size2
with torch.no_grad():
    for i,datat in enumerate(test):
        xt, yt = datat
        outmap, outt = model(xt)
    
        outt_c = outt[:,:,0]
        yt_c = yt
        for j in range(1,crop_n):
            outt_c = torch.cat((outt_c, outt[:,:,j]), 0)
            yt_c = torch.cat((yt_c, yt))
    
            _, predicted = torch.max(outt_c.data, 1)
            total += yt_c.size(0)
            correct += (predicted == yt_c).sum().item()
    
    
        _, predicted = torch.max(outt.data, 1)
        predicted = predicted.numpy()
        for j in range(batch_size2):
            count = np.bincount(predicted[j,:])
            a = np.argmax(count)
            b = np.max(count)
            for ci in range(len(count)):
                if count[ci]==b:
                    c = ci
            if a==yt.numpy()[j] and a==c:
                crop_c += 1
                    
        acc_test = crop_c/crop_t
        acc_val = correct / total
        print('Accuracy of the test: %.4f %%    Accuracy after vote%.4f %% ' % (100 * acc_val, 100*acc_test))

outmap = outmap.numpy()

aaa = outmap[3,:,:,2]
#aaa = aaa/np.max(abs(aaa))
plt.figure(dpi=300)
sns.heatmap(aaa, vmax= None ,vmin=None, cmap =  'coolwarm_r', mask=abs(outmap[2,:,:,0]) < 0.4).invert_yaxis() #

'''w_class = np.load('w_class.npy')
l2 = w_class[:,:,0,0]
plt.figure()
plt.plot(range(30),l2[3],l2[4])'''
'''plt.pcolormesh(range(35), np.array(range(5,65,2))/2, aaa,cmap =  'warmcool')
plt.colorbar()'''


'''label = [i for i in range(5)]  
maps = [i for i in range(5)]

maps[0] = np.concatenate((outmap[2],outmap[4], outmap[6]), axis=1)
maps[1] = np.concatenate((outmap[3],outmap[5], outmap[7]), axis=1)
maps[2] = np.concatenate((outmap[8],outmap[10], outmap[12]), axis=1)
maps[3] = np.concatenate((outmap[9],outmap[11], outmap[13]), axis=1)'''
'''maps_ = np.concatenate((outmap[0],outmap[1]), axis=1)
label[4] = np.concatenate((predicted[0],predicted[1]), axis=0)
maps[4] = np.mean(maps_[:,label[4]==0,:],1)

maps_ = np.concatenate((outmap[2],outmap[4], outmap[6]), axis=1)
label[0] = np.concatenate((predicted[2],predicted[4], predicted[6]), axis=0)
maps[0] = maps_[:,label[0]==1,:]

maps_ = np.concatenate((outmap[3],outmap[5], outmap[7]), axis=1)
label[1] = np.concatenate((predicted[3],predicted[5], predicted[7]), axis=0)
maps[1] = maps_[:,label[1]==2,:]

maps_ = np.concatenate((outmap[8],outmap[10], outmap[12]), axis=1)
label[2] = np.concatenate((predicted[8],predicted[10], predicted[12]), axis=0)
maps[2] = maps_[:,label[2]==3,:]

maps_ = np.concatenate((outmap[9],outmap[11], outmap[13]), axis=1)
label[3] = np.concatenate((predicted[9],predicted[11], predicted[13]), axis=0)
maps[3] = maps_[:,label[3]==4,:]'''

'''for k in range(4):
    plt.figure(dpi=600)
    for j in range(8):
        plt.subplot(4,2,j+1)
        for i in range(30):
            plt.scatter([i+2.5]*maps[k].shape[1], maps[k][i,:,j]-maps[4][i,j], s=1)#-maps[k][i,:,6]
        a = maps[k][:,:,j]-maps[4][:,j].reshape([30,1])#-maps[k][:,:,6]   
        plt.plot(np.array(range(2,32))+0.5, np.mean(a,1))'''
         
'''for k in range(4):
    plt.figure(dpi=600)
    plt.ylim(-10, 10)
    for j in range(8):
        plt.subplot(4,2,j+1)
        for i in range(30):
            plt.scatter([i+2.5]*maps[k].shape[1], maps[k][i,:,j], s=0.2)#-maps[k][i,:,6]
        a = maps[k][:,:,j]#-maps[k][:,:,6]   
        plt.plot(np.array(range(2,32))+0.5, np.mean(a,1))

for k in range(2):
    plt.figure(dpi=600)
    for j in range(8):
        plt.subplot(4,2,j+1)
        plt.boxplot(maps[k][:,:,j].T)#-maps[k][i,:,6]
        plt.ylim(-10, 10)'''

'''plt.figure(dpi=600)
for j in range(8):
    plt.subplot(4,2,j+1)
    for i in range(30):
        plt.scatter([i+2.5]*47, maps[2][i,:,j]-maps[3][i,:47,j], s=1)#-maps[k][i,:,6]
    a = maps[2][:,:,j]-maps[3][:,:47,j]#-maps[k][:,:,6]   
    plt.plot(np.array(range(2,32))+0.5, np.mean(a,1))'''        
         