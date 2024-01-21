# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:03:30 2020

@author: 77509
"""


import numpy as np

#from sklearn import preprocessing
#from sklearn.utils import shuffle
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
#from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)

def normal_value(data):
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    mean = mean.reshape([-1,1])
    std_m = np.eye(64)
    for i in range(64):
        std_m[i,i]=1/std[i]
    
    return mean, std_m



def create_trial(sub_list, m_t, length, n_test, kf, clas, m=0):
    ''' 静息态片段长度'''
    rest_n = 160
    min_t = (1-rest_n)/160
    if min_t>m_t:
        min_t = m_t
    
    max_t = (length - 1)/160+m_t
    if max_t >=4.1:
        max_t = 4.1
        m_t = 4.1-(length - 1)/160
    
    ''' 任务集片段长度 '''
    
    #n_test = 2 偶数，每个文件内找n_test个不同类的trial(每个文件有12个trail)
    data_sub_train = [[],[],[]]
    data_sub_test = [[],[],[]]
    for sub in sub_list:    
        if clas==5:
            raw_fnames = eegbci.load_data(sub, 1) 
            raw = read_raw_edf(raw_fnames[0], preload=True) #9760
            raw._data = raw._data*10e4
            raw.filter(1, 48., fir_design='firwin', skip_by_annotation='edge')
            task2D = raw._data
            task = np.ones([14,64,640])
            for i in range(1,15):
                task[i-1] = task2D[:,i*640:(i+1)*640]
            
            data_sub_test[0].append(task[2*kf:2*kf+2])
            data_sub_test[1].append(np.zeros(2))
        
            lei = [ k for k in range(14)]
            lei.pop(2*kf)
            lei.pop(2*kf)
    
            data_sub_train[0].append(task[lei])
            data_sub_train[1].append(np.zeros([12,1]))
        for file in [[4,8,12, 1,2],[6,10,14, 3,4]]:
            for f in file[:3]:
                raw_fnames = eegbci.load_data(sub, f)
                raw = read_raw_edf(raw_fnames[0], preload=True)
                raw._data = raw._data*10e4
                
                raw.filter(1, 48., fir_design='firwin', skip_by_annotation='edge')        
                
                event_id = dict(left=file[3], right=file[4])
                events, _ = events_from_annotations(raw, event_id=dict(T1=file[3], T2=file[4]))     
                picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,exclude='bads')
                        
                epochs = Epochs(raw, events, event_id, min_t, max_t, proj=True, picks=picks, baseline=None, preload=True)
                epochs_task = epochs.copy().crop(tmin=m_t, tmax=max_t)
                
                task = epochs_task.get_data()
                label = epochs.events[:,2:]
                
                '''if task.shape[0] == 15:
                    task, rest, label = task[2:-1], rest[2:-1], label[2:-1]
                elif task.shape[0] == 14:     
                    task, rest, label = task[2:], rest[2:], label[2:]
                else:     
                    task, rest, label = task[1:], rest[1:], label[1:] ''' 
    
                '''for i in range(task.shape[0]):
                    mean,std = normal_value(rest[i])
                    task[i] = np.matmul(std, (task[i]-mean))'''
                
                #task,label,rest = shuffle(task,label,rest)
                lei1 = [ k for k in range(task.shape[0]) if label[k]==file[3]]
                lei2 = [ k for k in range(task.shape[0]) if label[k]==file[4]]                
                
                data_sub_test[0].append( np.concatenate((task[lei1[kf]].reshape([1,64,task.shape[2]]),task[lei2[kf]].reshape([1,64,task.shape[2]])), axis=0))
                data_sub_test[1].append( np.concatenate((label[lei1[kf]],label[lei2[kf]]), axis=0))
               
                lei1.pop(kf)
                lei2.pop(kf)
                data_sub_train[0].append( np.concatenate((task[lei1],task[lei2]), axis=0))
                data_sub_train[1].append( np.concatenate((label[lei1],label[lei2]), axis=0))
        
        
    train_x = np.concatenate((data_sub_train[0][:]), axis=0)
    train_y = np.concatenate((data_sub_train[1][:]), axis=0)
    
    test_x = np.concatenate((data_sub_test[0][:]), axis=0)
    test_y = np.concatenate((data_sub_test[1][:]), axis=0)
    
    
    
    
    if m == 0:
        return train_x,train_y.squeeze(),test_x,test_y
    else:     
        sub_train = np.repeat(np.arange(m), 78).reshape([-1,1])
        sub_test = np.repeat(np.arange(m), 12).reshape([-1,1])
        return train_x,train_y,sub_train, test_x,test_y, sub_test


def create_trial2(sub_list, m_t, length, n_test, kf, clas, m=0):
    ''' 静息态片段长度'''
    rest_n = 190
    min_t = -rest_n/160
    if min_t>m_t:
        min_t = m_t
    
    max_t = (length - 1)/160+m_t

    
    ''' 任务集片段长度 '''
    
    #n_test = 2 偶数，每个文件内找n_test个不同类的trial(每个文件有12个trail)
    data_sub_train = [[],[],[]]
    data_sub_test = [[],[],[]]
    for sub in sub_list:    
        if clas==5:
            raw_fnames = eegbci.load_data(sub, 1) 
            raw = read_raw_edf(raw_fnames[0], preload=True) #9760
            raw._data = raw._data*10e4
            raw.filter(0.1, 48., fir_design='firwin', skip_by_annotation='edge')
            task2D = raw._data
            task = np.ones([14,64,640+190])
            for i in range(1,15):
                task[i-1] = task2D[:,i*320:i*320+640+190]
            
            data_sub_test[0].append(task[2*kf:2*kf+2])
            data_sub_test[1].append(np.zeros(2))
        
            lei = [ k for k in range(14)]
            lei.pop(2*kf)
            lei.pop(2*kf)
    
            data_sub_train[0].append(task[lei])
            data_sub_train[1].append(np.zeros([12,1]))
        for file in [[4,8,12, 1,2],[6,10,14, 3,4]]:
            for f in file[:3]:
                raw_fnames = eegbci.load_data(sub, f)
                raw = read_raw_edf(raw_fnames[0], preload=True)
                raw._data = raw._data*10e4
                
                raw.filter(0.1, 48., fir_design='firwin', skip_by_annotation='edge')        
                
                event_id = dict(left=file[3], right=file[4])
                events, _ = events_from_annotations(raw, event_id=dict(T1=file[3], T2=file[4]))     
                picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,exclude='bads')
                        
                epochs = Epochs(raw, events, event_id, min_t, max_t, proj=True, picks=picks, baseline=None, preload=True)
                #epochs_task = epochs.copy().crop(tmin=m_t, tmax=max_t)
                
                task = epochs.get_data()
                label = epochs.events[:,2:]
                
                '''if task.shape[0] == 15:
                    task, rest, label = task[2:-1], rest[2:-1], label[2:-1]
                elif task.shape[0] == 14:     
                    task, rest, label = task[2:], rest[2:], label[2:]
                else:     
                    task, rest, label = task[1:], rest[1:], label[1:] ''' 
    
                '''for i in range(task.shape[0]):
                    mean,std = normal_value(rest[i])
                    task[i] = np.matmul(std, (task[i]-mean))'''
                
                #task,label,rest = shuffle(task,label,rest)
                lei1 = [ k for k in range(task.shape[0]) if label[k]==file[3]]
                lei2 = [ k for k in range(task.shape[0]) if label[k]==file[4]]                
                
                data_sub_test[0].append( np.concatenate((task[lei1[kf]].reshape([1,64,task.shape[2]]),task[lei2[kf]].reshape([1,64,task.shape[2]])), axis=0))
                data_sub_test[1].append( np.concatenate((label[lei1[kf]],label[lei2[kf]]), axis=0))
               
                lei1.pop(kf)
                lei2.pop(kf)
                data_sub_train[0].append( np.concatenate((task[lei1],task[lei2]), axis=0))
                data_sub_train[1].append( np.concatenate((label[lei1],label[lei2]), axis=0))
        
        
    train_x = np.concatenate((data_sub_train[0][:]), axis=0)
    train_y = np.concatenate((data_sub_train[1][:]), axis=0)
    
    test_x = np.concatenate((data_sub_test[0][:]), axis=0)
    test_y = np.concatenate((data_sub_test[1][:]), axis=0)
    
    
    
    
    if m == 0:
        return train_x,train_y.squeeze(),test_x,test_y
    else:     
        sub_train = np.repeat(np.arange(m), 78).reshape([-1,1])
        sub_test = np.repeat(np.arange(m), 12).reshape([-1,1])
        return train_x,train_y,sub_train, test_x,test_y, sub_test
    
if  __name__ == '__main__':
    sub_list = [1,7]
    train_x,train_y,  test_x,test_y = create_trial(sub_list=sub_list, m_t=0, length = 640, n_test=2, kf=0, clas=5, m=0)

a = 10e5