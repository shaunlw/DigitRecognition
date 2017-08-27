#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 22:01:06 2017

@author: xlw
"""

import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    label_path = os.path.join(path,'%s-labels-idx1-ubyte' % kind)
    image_path = os.path.join(path,'%s-images-idx3-ubyte' % kind)
    with open(label_path,'rb') as lb:
        magic, n = struct.unpack('>II',lb.read(8)) # '>' means big endian, I = unsigned int, II = 8 byte; read 8 bytes and upack into 2 x 4 bytes. The first 8 bytes are metadata that we don't need. see website data description  
        labels = np.fromfile(lb,dtype=np.uint8) #8-bit unsigned int, read the rest into 1-d array
    with open(image_path,'rb') as img:
        magic,num,rows,cols = struct.unpack('>IIII',img.read(16)) # unpack the first 16 bytes into 4x4 blocks. num=how many images=6000, rows*cols=28*28 are pixels per image. 
        imgs = np.fromfile(img, dtype=np.uint8).reshape(len(labels),784) # 28*28 = 784
    return imgs,labels

def get_train_test():
    cur_path = os.path.join('/Users/xiangliwang/Python/ml/Raschka/code/digit.recognition','mnist')
    print(cur_path)
    X_train, y_train = load_mnist(cur_path,kind='train')
    print('row: %d, col: %d' % (X_train.shape[0], X_train.shape[1]))
    X_test, y_test = load_mnist(cur_path, kind='t10k')
    print('row: %d, col: %d' % (X_test.shape[0], X_test.shape[1]))
    return X_train, X_test, y_train, y_test

#plot some example images
def plot_example():
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,5,sharex=True,sharey=True)
    ax=ax.ravel()
    
    X_train, X_test, y_train, y_test = get_train_test()
    for i in range(10): #plot 0 to 9
        img = X_train[y_train==i][0].reshape(28,28) # extract the first of the training sampled labeld as i
        ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    
    #plot different versions of the same digit    
    fig,ax=plt.subplots(5,5,sharex=True,sharey=True)
    ax=ax.ravel()
    for i in range(25):
        img = X_train[y_train==7][i].reshape(28,28)
        ax[i].imshow(img,cmap='Greys',interpolation='nearest')