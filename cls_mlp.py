#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 23:37:17 2017

@author: xlw
"""

import numpy as np
from scipy.special import expit
import sys
'''three layer perceptron network'''

class NeuralNetMLP(object):
    def __init__(self, n_output,n_features,n_hidden=30,l1=0,l2=0,
                 epochs=500,eta=.001,alpha=0,decrease_const=0,shuffle=True,
                 minibatches=1,random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches
        
    def _encode_labels(self,y,k):
        onehot = np.zeros((k,y.shape[0]))
        for idx,val in enumerate(y):
            onehot[val,idx] = 1.0
        return onehot
        
    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0,1.0,size=self.n_hidden*(self.n_features + 1))
        w1 = w1.reshape(self.n_hidden,self.n_features+1) # shape[n_hidden,n_feature],for each hidden unit you need one weight vector
        w2 = np.random.uniform(-1.0,1.0,size=self.n_output*(self.n_hidden + 1))
        w2 = w2.reshape(self.n_output,self.n_hidden+1)
        return w1, w2
        
    def _sigmoid(self,z): #phi() activation function returns activations
        return expit(z)
        
    def _sigmoid_gradient(self, z): #Raschka p64, derivation of cost function, & p370
        sg = self._sigmoid(z)
        return sg * (1 - sg)
        
    def _add_bias_unit(self, X, how='column'): # if feature is in column
        if how == 'column':
            X_new = np.ones((X.shape[0],X.shape[1] + 1)) # if you want 2-d array you have to put params in a tuple
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1,X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new
    
    def _feedforward(self,X,w1,w2):
        a1 = self._add_bias_unit(X, how = 'column') # equivalent to X_new, shape[n_sample,n_feature+1]
        z2 = w1.dot(a1.T) # shape[n_hidden, n_sample], net input for each sample
        a2 = self._sigmoid(z2) # activiation on layer2
        a2 = self._add_bias_unit(a2,how='row')
        z3 = w2.dot(a2) # net input for each output unit. w2.shape[n_output,n_hidden] dot a2.shape[n_hidden,n_sample] = shape[n_output,n_sample]
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3
    
    def _L2_reg(self,lambda_,w1,w2): # weights of layer 1 and 2 for all samples, w1 and w2 have different dimensionality [n_feature, n_hidden+1] vs [n_hidden+1, n_output]
          return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:,1:] ** 2)) #take square and then sum, vectorized computation
    
    def _L1_reg(self, lambda_,w1,w2):
        return (lambda_/2)*(np.abs(w1[:,1:]).sum() + np.abs(w2[:,1:]).sum()) # take abs and then sum, vectorized computation
        
    def _get_cost(self,y_enc,output,w1,w2):
        term1 = -y_enc*np.log(output)
        term2 = (1-y_enc)*np.log(1-output)
        cost = np.sum(term1 - term2)
        l1_term = self._L1_reg(self.l1,w1,w2) # although _L1_reg() is defined with four params, only three are neededtto call it, with self._L2_reg() though
        l2_term = self._L2_reg(self.l2,w1,w2) 
        cost += l1_term + l2_term
        return cost
        
    def _get_gradient(self,a1,a2,a3,z2,y_enc,w1,w2):
        #backpropagation
        sigma3= a3 - y_enc
        z2 = self._add_bias_unit(z2,how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:,:] # first row is bias factor
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)
        
        #regularize
        grad1[:,1:] += w1[:,1:] * (self.l1 + self.l2)
        grad2[:,1:] += w2[:,1:] * (self.l1 + self.l2)
        
        return grad1,grad2
        
    def predict(self,X):
        a1,z2,a2,z3,a3 = self._feedforward(X,self.w1,self.w2)
        y_pred = np.argmax(z3,axis=0)
        return y_pred
        
    def fit(self, X, y, print_progress=False):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output) #shape[n_sample,n_class]
        
        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)
        
        for i  in range(self.epochs):
            self.eta /= (1+self.decrease_const*i)
            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()
            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx],y_data[idx]
            mini = np.array_split(range(y_data.shape[0]), y_data[idx])
            
            for idx in mini:
                #feedforward
                a1, z2, a2, z3, a3 = self._feedforward(X[idx],self.w1,self.w2)
                cost = self._get_cost(y_enc=y_enc[:,idx], output=a3, w1=self.w1, w2=self.w2)
                self.cost_.append(cost)
                
                #compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1,a2=a2,a3=a3,z2=z2,y_enc=y_enc[:, idx], w1=self.w1,w2=self.w2)
                
                #update weights
                delta_w1 = self.eta * grad1 + self.alpha * delta_w1_prev # momentum based on previous learning speed
                delta_w2 = self.eta * grad2 + self.alpha * delta_w2_prev # learning speed depends on eta, gradient, previous learning experience
                self.w1 -= delta_w1
                self.w2 -= delta_w2 
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
        return self
        
    def _gradient_check(self,X,y,y_enc,w1,w2,epsilon,grad1,grad2):
        '''very slow, for debugging only
        returns:
            relative error: float, between numerically approximated gradients and the backpropagated gradients
        '''
        
        num_grad1 = np.zeros(np.shape(w1)) # shape[n_sample,n_feature]
        epsilon_arr1 = np.zeros(np.hape(w1))
        for i in range(w1.shape[0]):
            for j in range(w1.shape[1]):
                epsilon_arr1[i,j] = epsilon
                a1,z2,a2,z3,a3 = self._feedforward(X,
                                                   w1 - epsilon_arr1,
                                                   w2)
                cost1 = self._get_cost(y_enc, a3,
                                      w1 - epsilon_arr1,w2)
                a1,z2,a2,z3,a3 = self._feedforward(X,
                                                   w1 + epsilon_arr1,
                                                   w2)
                cost2 = self._get_cost(y_enc, a3,
                                      w1 + epsilon_arr1,w2)
                num_grad1[i,j] = (cost2 - cost1)/ (2 * epsilon)
                epsilon_arr1[i,j] = 0
                
        num_grad2 = np.zeros(np.shape(w2))
        epsilon_arr2 = np.zeros(np.shape(w2))
        for i in range(w2.shape[0]):
            for j in range(w2.shape[1]):
                epsilon_arr2[i,j] = epsilon
                a1,z2,a2,z3,a3 = self._feedforward(X,
                                                   w1,
                                                   w2 - epsilon_arr2)
                cost1 = self._get_cost(y_enc, a3,
                                      w1,w2 - epsilon_arr2)
                a1,z2,a2,z3,a3 = self._feedforward(X,
                                                   w1,
                                                   w2 + epsilon_arr2)
                cost2 = self._get_cost(y_enc, a3,
                                      w1,w2 + epsilon_arr2)
                num_grad2[i,j] = (cost2 - cost1)/ (2 * epsilon)
                epsilon_arr2[i,j] = 0         
        num_grad = np.hstack((num_grad1.flatten(),
                              num_grad2.flatten()))
        grad = np.hstack((grad1.flatten(),grad2.flatten()))
        norm1 = np.linalg.norm(num_grad - grad)
        norm2 = np.linalg.norm(num_grad)
        norm3 = np.linalg.norm(grad)
        relative_error= norm1/(norm2+norm3)
        return relative_error
        
        

            