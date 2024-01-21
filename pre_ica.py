# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:03:30 2020

@author: 77509
"""


import numpy as np

from sklearn import preprocessing
from sklearn.utils import shuffle
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)
import matplotlib.pyplot as plt



def normal_value(data):
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    mean = mean.reshape([-1,1])
    std_m = np.eye(64)
    for i in range(64):
        std_m[i,i]=1/std[i]
    
    return mean, std_m


mapping = {
    'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCz', 'Fc2.': 'FC2',
    'Fc4.': 'FC4', 'Fc6.': 'FC6', 'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1',
    'Cz..': 'Cz', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6', 'Cp5.': 'CP5',
    'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPz', 'Cp2.': 'CP2', 'Cp4.': 'CP4',
    'Cp6.': 'CP6', 'Fp1.': 'Fp1', 'Fpz.': 'Fpz', 'Fp2.': 'Fp2', 'Af7.': 'AF7',
    'Af3.': 'AF3', 'Afz.': 'AFz', 'Af4.': 'AF4', 'Af8.': 'AF8', 'F7..': 'F7',
    'F5..': 'F5', 'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'Fz', 'F2..': 'F2',
    'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8', 'Ft7.': 'FT7', 'Ft8.': 'FT8',
    'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10', 'Tp7.': 'TP7',
    'Tp8.': 'TP8', 'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1',
    'Pz..': 'Pz', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6', 'P8..': 'P8',
    'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POz', 'Po4.': 'PO4', 'Po8.': 'PO8',
    'O1..': 'O1', 'Oz..': 'Oz', 'O2..': 'O2', 'Iz..': 'Iz'}
 
    
fnames = eegbci.load_data(1, [4,8,12,6,10,14])
raws = [read_raw_edf(f, preload=True) for f in fnames]
raw = concatenate_raws(raws)
    
raw.rename_channels(mapping)
raw.set_montage('standard_1005')

event_id = dict(left=1)
events, _ = events_from_annotations(raw, event_id=dict(T1=1))     
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,exclude='bads')
epochs = Epochs(raw, events, event_id, -1, 4, proj=True, picks=picks, baseline=(None,0), preload=True)
epochs_task = epochs.copy().crop(tmin=0, tmax=4)
        
ica = ICA(n_components=64, random_state=92, method = 'fastica') #{'fastica', 'infomax', 'extended-infomax', 'picard'}

ica.fit(epochs_task)        
   
#raw.plot(order=picks, n_channels=64)           
#eog_inds, eog_scores = ica.find_bads_eog (raw, ch_name='Fpz')#, threshold=0.99
'''for i in range(4):  
    ica.plot_sources(raw, picks = np.arange(i*16,i*16+16))'''
#ica.mixing_matrix_ = ica.unmixing_matrix_.T
ica.plot_components()
ica.plot_properties(epochs, picks=[7])

m = np.matmul(ica.mixing_matrix_.T, ica.pca_components_ )
ica.mixing_matrix_ = m.T
ica.pca_components_ = np.eye(64)
ica.plot_components()
w_informax = np.linalg.inv(ica.mixing_matrix_)
np.save('w_informax.npy', w_informax)


#b = np.load('bp.npy')
'''ww = mixing_matrix(0.2, -0.4, al = 0.8)
w1 = ww'''

w = np.load('w_ica.npy')
w1 = np.zeros([64,64])
for i in range(w.shape[0]):
    w1[i,:] = w[i,0,0,:]
print(np.max(w1), np.min(w1))
#w1 = np.linalg.inv(w1)
print(np.max(w1), np.min(w1))
#w1 = preprocessing.scale(w1, axis=0)


ica.mixing_matrix_ = w1.T
ica.pca_components_ = np.eye(64)
ica.plot_components()

#ica.plot_properties(raw, picks=[61])
'''matrix = mixing_matrix(0.2, -0.2)
ica.mixing_matrix_ = matrix.T
ica.plot_components()'''


