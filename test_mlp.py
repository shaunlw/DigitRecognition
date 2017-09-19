#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:57:27 2017

@author: xlw
"""

from cls_mlp import NeuralNetMLP
import data_mungle as dm
import numpy as np

X_train, X_test, y_train, y_test = dm.get_train_test()

nn = NeuralNetMLP(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=10,
                  l2=.1,
                  l1=.0,
                  epochs=2,
                  eta=.001,
                  alpha=.001,
                  decrease_const=.00001,
                  shuffle=True,
                  minibatches=50,
                  random_state=1)
nn.fit(X_train,y_train,print_progress=True)

import matplotlib.pyplot as plt
plt.plot(range(len(nn.cost_)),nn.cost_)
plt.ylim([0,1000])
plt.xlim([0,5000])
plt.xlabel('number of run')
plt.ylabel('cost')

y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('\n',acc)
