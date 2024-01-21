# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:32:24 2020

@author: 77509
"""


import numpy as np
import math as m
import scipy.signal as signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def FFT(Fs, data):
    L = len(data)  # 信号长度
    N = np.power(2, np.ceil(np.log2(L))) * 4  # 下一个最近二次幂
    N = N.astype('int')
    FFT_y = (fft(data, N)) / L * 2  # N点FFT，但除以实际信号长度 L
    Fre = np.arange(int(N / 2)) * Fs / N  # 频率坐标
    FFT_y = FFT_y[range(int(N / 2))]  # 取一半

    mode = abs(FFT_y)
    ang = np.angle(FFT_y)
    return Fre, mode, ang


def myfilter(fs, fl, fh, fliter_n, mode = 'N'):
    '''
        标准结构
        fs:采样率，单位Hz
        fl:下限截止频率，单位Hz
        fh:上限截止频率，单位Hz
        fliter_n:阶数，输出长度
        mode:加窗类型，默认矩形窗
                              '''

    wl = fl/fs*2*m.pi
    wh = fh/fs*2*m.pi
    
    fliter_d = (fliter_n-1)/2
    fliter_a = np.zeros(fliter_n)
    Rhm = Rhn = Rt = np.zeros(fliter_n)
    
    for i in range(fliter_n):
        if i == fliter_d:
            fliter_a[i] = wh - wl
        else:
            fliter_a[i] = m.sin(wh*(i-fliter_d))/(m.pi*(i-fliter_d)) - m.sin(wl*(i-fliter_d))/(m.pi*(i-fliter_d))

        Rhn[i] = 0.5-0.5*m.cos(2*i*m.pi/(fliter_n-1))    
        Rhm[i] = 0.54-0.46*m.cos(2*i*m.pi/(fliter_n-1))
        
        if i<= (fliter_n-1)/2:
            Rt[i] = 2*i/(fliter_n-1)
        else:
            Rt[i] = 2 - 2*i/(fliter_n-1)
            
    fliter1 = fliter_a
    fliter2 = fliter_a * Rt
    fliter2 = fliter2[1:-1]
    fliter3 = fliter_a * Rhn
    fliter3 = fliter3[1:-1]
    fliter4 = fliter_a * Rhm
    fliter4 = fliter4[1:-1]

    '''t = np.arange(0, 1, 1/160)
    x= signal.chirp(t, f0=0.1, t1 = 1, f1=fh*2)
    
    N = fliter_n - 3
    plt.figure()
    fre,fft_y = FFT(160, x)
    plt.plot(fre,fft_y)
    
    plt.figure()
    plt.subplot(2,2,1)
    x_f = np.zeros(160-N)
    for i in range(160-N):
        x_f[i] = np.matmul(x[i:i+N+1],fliter1)
    fre,fft_y = FFT(160, x_f)
    plt.plot(fre,fft_y)
    
    plt.subplot(2,2,2)
    x_f = np.zeros(160-N)
    for i in range(160-N):
        x_f[i] = np.matmul(x[i:i+N+1],fliter2)
    fre,fft_y = FFT(160, x_f)
    plt.plot(fre,fft_y)
    
    plt.subplot(2,2,3)
    x_f = np.zeros(160-N)
    for i in range(160-N):
        x_f[i] = np.matmul(x[i:i+N+1],fliter3)
    fre,fft_y = FFT(160, x_f)
    plt.plot(fre,fft_y)
    
    plt.subplot(2,2,4)
    x_f = np.zeros(160-N)
    for i in range(160-N):
        x_f[i] = np.matmul(x[i:i+N+1],fliter4)
    fre,fft_y = FFT(160, x_f)
    plt.plot(fre,fft_y)'''

    if mode == 'N':
        return fliter1
    elif mode == 't':
        return fliter2
    elif mode == 'hn':
        return fliter3
    else:
        return fliter4


def myfilter_double(fs, fl, fh, fliter_n, mode):
    '''级联结构'''
    fs = 160
    fl = 20
    fh = 40
    fliter_n = 9

    wl = fl / fs * 2 * m.pi
    wh = fh / fs * 2 * m.pi

    fliter_n = fliter_n + 2
    fliter_d = (fliter_n - 1) / 2
    fliter1 = fliter2 = np.zeros(fliter_n)

    for i in range(fliter_n):
        if i == fliter_d:
            fliter1[i] = wh
            fliter2[i] = wl
        else:
            fliter1[i] = m.sin(wh * (i - fliter_d)) / (m.pi * (i - fliter_d))
            fliter2[i] = m.sin(m.pi * (i - fliter_d)) / (m.pi * (i - fliter_d)) - m.sin(wl * (i - fliter_d)) / (
                        m.pi * (i - fliter_d))

    t = np.arange(0, 1, 1 / 160)
    x = signal.chirp(t, f0=0.1, t1=1, f1=fh * 2)

    N = fliter_n - 1
    plt.figure()
    fre, fft_y = FFT(160, x)
    plt.plot(fre, fft_y)

    plt.figure()
    x_f1 = np.zeros(160 - N)
    x_f2 = np.zeros(160 - 2 * N)
    for i in range(160 - N):
        x_f1[i] = np.matmul(x[i:i + N + 1], fliter1)
    for i in range(160 - 2 * N):
        x_f2[i] = np.matmul(x_f1[i:i + N + 1], fliter2)

    fre, fft_y = FFT(160, x_f2)
    plt.plot(fre, fft_y)



if  __name__ == '__main__':
    filter = myfilter(fs=160, fl=20, fh=40, fliter_n=21, mode ='N')
    Fre, mode, ang = FFT(160, filter)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(Fre, mode)
    plt.subplot(1, 2, 2)
    plt.plot(Fre, ang)
    plt.show()






