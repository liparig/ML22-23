#!/usr/bin/env python
# coding: utf-8
import numpy as np

from costants import SIGMOID, TANH, RELU, IDENTITY

def identity(x):
    return x


def d_identity(x):
    return 1

def sigmoid(net):
    
    """sigmoid_m=[]
    
    #avoid overflow
    for x in net.flatten():
        if x<0:
            sigmoid_m.append(np.exp(x)/(1+np.exp(x)))
        else:
            sigmoid_m.append(1 / (1 + np.exp(-x)))
            
    sigmoid_m=np.array(sigmoid_m)
    sigmoid_m=sigmoid_m.reshape(net.shape)"""
    return np.where(net >= 0, 1/(1 + np.exp(-net)), np.exp(net)/(1 + np.exp(net)))


def d_sigmoid(net):
    return np.multiply(sigmoid(net) , np.subtract( 1 , sigmoid(net)))


def tanh(net):
    return np.tanh(net)


def d_tanh(net):
    return np.subtract(1,np.power(np.tanh(net), 2)
    )

def relu(net):
    return np.maximum(net, 0)

def d_relu(net):
    net[net <= 0] = 0
    net[net > 0] = 1
    return net

activations = {
    SIGMOID: sigmoid,
    TANH: tanh,
    RELU: relu,
    IDENTITY:identity,
}
derivatives = {
    SIGMOID: d_sigmoid,
    TANH: d_tanh,
    RELU: d_relu,
    IDENTITY:d_identity
}

