#!/usr/bin/env python
# coding: utf-8
import numpy as np

def identity(x):
    return x


def d_identity(x):
    return 1

def sigmoid(net):
    sigmoid_m=[]
    #avoid overflow
    for x in net.flatten():
        if x<0:
            sigmoid_m.append(np.exp(x)/(1+np.exp(x)))
        else:
            sigmoid_m.append(1 / (1 + np.exp(-x)))
    sigmoid_m= np.array(sigmoid_m)
    sigmoid_m=sigmoid_m.reshape(net.shape)
    return sigmoid_m


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
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'identity':identity,
}
derivatives = {
    'sigmoid': d_sigmoid,
    'tanh': d_tanh,
    'relu': d_relu,
    'identity':d_identity
}

