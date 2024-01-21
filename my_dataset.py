# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:00:33 2020

@author: 77509
"""


from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def mydataset(path, batch_size, test_size):
    '''
            验证集与训练集有重叠
                                      '''
    xnp = np.load(path)
    xnp = xnp.astype('float32')
    ypath = path.replace("x", "y")
    ynp = np.load(ypath)
    ynp = ynp.squeeze()
    
    x_train,x_test,y_train,y_test = train_test_split(xnp, ynp, test_size=test_size, random_state=0)

    
    X_train = torch.from_numpy(x_train)
    X_train = X_train.unsqueeze(3)
    Y_train = torch.from_numpy(y_train).long()
    dataset1 = TensorDataset(X_train,Y_train)
    data_train = DataLoader(dataset = dataset1, batch_size = batch_size, shuffle = True)
    
    X_test = torch.from_numpy(x_test)
    X_test = X_test.unsqueeze(3)
    Y_test = torch.from_numpy(y_test).long()
    dataset2 = TensorDataset(X_test,Y_test)
    data_test = DataLoader(dataset = dataset2, batch_size = batch_size, shuffle = True)
    return data_train, data_test


def train_test(path, batch_size, a):
    '''
            验证集与训练集无重叠
                                      '''
    xnp = np.load(path)
    xnp = xnp.astype('float32')
    ypath = path.replace("x", "y")
    ynp = np.load(ypath)
    ynp = ynp.squeeze()
    
    X_train = torch.from_numpy(xnp)
    X_train = X_train.unsqueeze(3)
    Y_train = torch.from_numpy(ynp).long()
    
    dataset1 = TensorDataset(X_train,Y_train)
    data_train = DataLoader(dataset = dataset1, batch_size = batch_size, shuffle = a)
    return data_train

"*"
def data_nopath(xnp, ynp, batch_size, a):
    '''
            验证集与训练集无重叠
                                      '''
    xnp = xnp.astype('float32')
    ynp = ynp.squeeze()
    
    X_train = torch.from_numpy(xnp)
    X_train = X_train.unsqueeze(3)
    Y_train = torch.from_numpy(ynp).long()
    
    dataset1 = TensorDataset(X_train,Y_train)
    data_train = DataLoader(dataset = dataset1, batch_size = batch_size, shuffle = a)
    return data_train

def data_nopath_m(xnp, ynp1, ynp2, batch_size, a):
    '''
            验证集与训练集无重叠
                                      '''
    xnp = xnp.astype('float32')
    ynp1 = ynp1.squeeze()
    ynp2 = ynp2.squeeze()
    
    X_train = torch.from_numpy(xnp)
    X_train = X_train.unsqueeze(3)
    Y_train1 = torch.from_numpy(ynp1).long()
    Y_train2 = torch.from_numpy(ynp2).long()
    
    dataset = TensorDataset(X_train,Y_train1, Y_train2)
    data_train = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = a)
    return data_train
'''path = 'eegbci/eegbci1/trainx1 7 29 25 31.npy'

data_loader = train_test(path, 20)

 
for i, data in enumerate(data_loader):
    #print(i)
    x, y = data
    #print(x.dtype, y.dtype)
    if i <=20:
        print(y.data)'''
    