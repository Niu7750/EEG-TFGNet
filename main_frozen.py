# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:06:04 2020

@author: 77509
"""
from pre_trail import create_trial
#from pre_trail_noeog import create_trial
from trialNet import ica_erdsNet1
from my_dataset import data_nopath
import matplotlib.pyplot as plt

import numpy as np
from myfilter import myfilter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.fftpack import fft
import time
import os


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

for s in range(1):
    chart = np.zeros([7,10])
    sub_list = [1]

    for ii in range(10):
        for kf in range(6,7):
            clas = 5
            crop_l = 32
            trail_l = 640 #560
            poolmax_stride = 1

            crop_n = int((trail_l - crop_l)/poolmax_stride + 1)
            ff = []
            for i in range(2,32,2):
                ff.append([i,i+2])
            filter_list = ff

            #filter_list = [[4,6], [12,14], [15,17], [21,23]]
            #1/[[2,4],[4,6],[6,8],[8,10],[14,16],[16,18],[18,20],[22,24],[28,30],[30,32]]
            #2/0.69/[[2,4],[4,6],[6,8],[8,10],[12,14],[14,16],[16,18],[18,20],[20,22],[22,24],[24,26]]
            #3/0.72/[[4,8],[12,14],[14,18],[20,25],[30,36],[40,48]]
            #[[2,4],[4,6],[6,8],[8,10],[10,12],[12,14],[14,16],[16,18],[18,20],[20,22],[22,24],[24,26],[26,28],[28,30]]
            #2/ [[3,5],[11,13],[16,18],[22,24],[23,25],[29,31]]
            #1/ [[12, 14], [18, 20], [28, 30]]
            #3/ [[1,3],[9,11],[29,31],[34,36]]

            filters_time_size = 31

            batch_size1 = 4
            batch_size2 = 14

            train_x,train_y,test_x,test_y = create_trial(sub_list=sub_list, m_t=0.5, length = trail_l, n_test=2, kf=kf, clas=clas, m=0)
            rest = torch.tensor(train_x[-14:].astype('float32'))
            rest = rest.unsqueeze(3)

            '''2class change'''
            '''train_x = train_x[(train_y[:,0] > 2)]
            train_y = train_y[(train_y > 2)]-3
            test_x = test_x[(test_y > 2)]
            test_y = test_y[(test_y > 2)] - 3'''

            train = data_nopath(train_x,train_y, batch_size1, True)
            test = data_nopath(test_x,test_y, batch_size2, False)

            #model = torch.load('model.pth')
            model = ica_erdsNet1(64, clas, input_time_length=trail_l, crop_length = crop_l,
                                n_filters_ica = 16,
                                filters = filter_list,
                                filters_time_size = filters_time_size ,
                                poolmax_size=80,poolmax_stride=poolmax_stride,
                                drop_prob1=0.5)

            initial = model.conv_ica.weight.data.numpy()



            '''判断输出尺寸是否正确'''
            out_pre, loss_f, er = model(torch.ones(batch_size1, 64, trail_l, 1))
            size = out_pre.data.size()
            print('输 出 尺 寸： ', size)
            print('输 出 尺 寸 是 否 正 确： ', size[-1]==crop_n)


            ''' 训练 '''
            criterion = nn.CrossEntropyLoss()
            #optimizer = optim.Adam( model.parameters(),  lr=0.001, weight_decay=1e-4, )#,{'params': model.parameters()}

            '''optimizer = optim.Adam([ {'params':model.conv_ica.parameters(), 'lr':1e-3, 'weight_decay':1e-4},
                                    {'params':model.conv_time.parameters(), 'lr':0},
                                    {'params':model.conv_class.parameters(), 'lr':1e-3,'weight_decay':1e-4}],)'''
            model.conv_time.weight.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.9, 0.999),
                                   eps=1e-08, weight_decay=1e-4)

            C =[]
            CC = []
            time_start = time.time() #开始计时

            for epoch in range(600):
                acc_all = []
                running_loss=0.0
                correct = 0
                total = 0

                #model.dropout = True
                for i,data in enumerate(train):
                    x, y = data
                    #y = y-1

                    out, out_all, energy = model(x)
                    #print(out.data.size())
                    #print(out_filter.shape)

                    out_c = out[:,:,0]
                    y_c = y
                    for j in range(1,crop_n):
                        out_c = torch.cat((out_c, out[:,:,j]), 0)
                        y_c = torch.cat((y_c, y))


                    loss = criterion(out_c, y_c)

                    '''train_filter = model.conv_time.weight.data.squeeze()
                    train_filter = torch.abs(train_filter)
                    loss_f = torch.mean(torch.abs(train_filter - train_filter[:, torch.arange(train_filter.size(1) - 1, -1, -1).long()]))
                    loss_f = loss_f.requires_grad_()
                    loss_f10 = loss_f*1000000'''

                    #loss2.backward(retain_graph=True)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    '''loss_f10.backward()
                    optimizer2.step()
                    optimizer2.zero_grad()'''


                    _, predicted = torch.max(out_c.data, 1)
                    correct_b = (predicted == y_c).sum().item()

                    acc_b = correct_b/y_c.size(0)
                    acc_all.append(acc_b)

                    correct += correct_b
                    total += y_c.size(0)
                    running_loss += loss.item()

                    pack = 5
                    if i % pack == pack-1: # print every 2000 mini-batches
                        #print(loss, loss_f)
                        #print(energy, loss, loss2)
                        acc_train = 100 * correct / total
                        print('[%d, %5d] loss: %.7f acc: %.2f %%' % (epoch + 1, i + 1, running_loss / pack, 100 * correct / total))
                        running_loss = 0.0
                        correct = 0
                        total = 0

                correct = 0
                total = 0
                crop_c = 0
                crop_t = batch_size2
                #model.dropout = False
                with torch.no_grad():
                    for i,datat in enumerate(test):
                        xt, yt = datat
                        #yt = yt-1
                        outt, loss_f,er = model(xt)

                        outt_c = outt[:,:,0]
                        yt_c = yt
                        for j in range(1,crop_n):
                            outt_c = torch.cat((outt_c, outt[:,:,j]), 0)
                            yt_c = torch.cat((yt_c, yt))

                        _, predicted = torch.max(outt_c.data, 1)
                        p1 = predicted.numpy()
                        p11 = yt_c.numpy()
                        total += yt_c.size(0)
                        correct += (predicted == yt_c).sum().item()

                        _, predicted = torch.max(outt.data, 1)
                        predicted = predicted.numpy()
                        p2 = np.zeros(batch_size2)
                        p22 = yt.numpy()
                        for j in range(batch_size2):
                            count = np.bincount(predicted[j, :])
                            a = np.argmax(count)
                            b = np.max(count)
                            for ci in range(len(count)):
                                if count[ci] == b:
                                    c = ci
                            p2[i] = (a + c) / 2
                            if a == yt.numpy()[j] and a == c:
                                crop_c += 1

                    acc_test = crop_c/crop_t
                    acc_val = correct / total

                    C.append(acc_val)
                    CC.append(acc_test)
                    if acc_test >= 1:
                        break

                    elif acc_val >= 0.6:
                        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001,
                                               betas=(0.9, 0.999),
                                               eps=1e-08, weight_decay=1e-4)


                    print('Accuracy of the test: %.4f %%    Accuracy after vote%.4f %% ' % (100 * acc_val, 100*acc_test))

            time_end = time.time()  # 结束计时
            time_c = time_end - time_start  # 运行所花时间

            '''测试集准确率'''
            correct = 0
            total = 0
            crop_c = 0
            crop_t = batch_size2
            with torch.no_grad():
                    for i,datat in enumerate(test):
                        xt, yt = datat
                        #yt = yt-1
                        outt, loss_f, er = model(xt)

                        outt_c = outt[:,:,0]
                        yt_c = yt
                        for j in range(1,crop_n):
                            outt_c = torch.cat((outt_c, outt[:,:,j]), 0)
                            yt_c = torch.cat((yt_c, yt))

                        _, predicted = torch.max(outt_c.data, 1)
                        p1 = predicted.numpy()
                        p11 = yt_c.numpy()
                        total += yt_c.size(0)
                        correct += (predicted == yt_c).sum().item()


                        _, predicted = torch.max(outt.data, 1)
                        predicted = predicted.numpy()
                        p2 = np.zeros(batch_size2)
                        p22 = yt.numpy()
                        for j in range(batch_size2):
                            count = np.bincount(predicted[j,:])
                            a = np.argmax(count)
                            b = np.max(count)
                            for ci in range(len(count)):
                                if count[ci]==b:
                                    c = ci
                            p2[i]=(a+c)/2
                            if a==yt.numpy()[j] and a==c:
                                crop_c += 1
                    acc_test = crop_c/crop_t


                    acc_val = correct / total
                    print('Accuracy of the test: %.4f %%    Accuracy after vote%.4f %% ' % (100 * acc_val, 100*acc_test))



            cc = np.max(C)
            cn = np.argmax(C)
            cc1 = np.max(CC)
            cn1 = np.argmax(CC)
            print(cc, cn, time_c)
            print(cc1, cn1)

            chart[kf,ii] = cc1
            path = 'frozen_5分类（为kalman）/S%d/%df_8c_%d single/%d %.4f-%.4f-%d-%f(%f'%(sub_list[0],len(filter_list), crop_l, kf+1, cc, cc1, cn1,time_c, acc_test)
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model, path+'/model.pth')
            np.save(path+'/initial_ica', initial)

            ''' 
                   可视化 
                             '''
            '''混淆矩阵'''
            hun1= np.zeros([clas+1,clas+1])
            hun2 = np.zeros([clas+1,clas+1])
            for i in range(p1.shape[0]):
                hun1[int(p11[i]),int(p1[i])] = hun1[int(p11[i]),int(p1[i])]+1
            for i in range(p2.shape[0]):
                hun2[int(p22[i]),int(p2[i])] = hun2[int(p22[i]),int(p2[i])]+1
            for i in range(clas):
                hun1[clas,i] = hun1[i,i]/np.sum(hun1[:clas,i])
                hun1[i,clas] = hun1[i,i]/np.sum(hun1[i,:clas])
                hun2[clas,i] = hun2[i,i]/np.sum(hun2[:clas,i])
                hun2[i,clas] = hun2[i,i]/np.sum(hun2[i,:clas])

            np.save(path+'/hun', np.concatenate((hun1, hun2), axis=0))

            '''label = [0,0,1,2,1,2,1,2,3,4,3,4,3,4]
            #label = [0,1,0,1,0,1,2,3,2,3,2,3]
            plt.figure(dpi = 100, figsize=(5,5))
            plt.subplots_adjust(left=None, bottom=0.1, right=None, top=2,wspace=None, hspace=None)
            for i in range(batch_size2):
                plt.subplot(batch_size2,1,i+1)
                plt.ylim(-0.5, 4.5)
                x_major_locator=plt.MultipleLocator(1)
                y_major_locator=plt.MultipleLocator(1)
                ax=plt.gca()
                ax.xaxis.set_major_locator(x_major_locator)
                ax.yaxis.set_major_locator(y_major_locator)
                plt.scatter([k for k in range(crop_n)], predicted[i,:], s=10)
                plt.plot([-0.5,crop_n], [label[i], label[i]], color='r', linewidth=1)

            plt.savefig(path+'/vote.jpg', dpi=300, bbox_inches='tight')'''

            w_ica = model.conv_ica.weight.data.numpy()
            #np.save('w_ica', w_ica)
            np.save(path+'/w_ica', w_ica)
            w_class = model.conv_class.weight.data.numpy()
            np.save(path+'/w_class', w_class)

    np.save(path+'/chart.npy', chart)
