#!/usr/bin/env python
# coding: utf-8
import numpy as np

import costants as C

# there are 5 differents activation functions with their derivatives that they are: 
# SIGMOID 
# TANH
# RELU
# LEAKYRELU
# IDENTITY

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
            
    sigmoid_m=np.array(sigmoid_m)
    sigmoid_m=sigmoid_m.reshape(net.shape)
    return sigmoid_m


def d_sigmoid(net):
    return np.multiply(sigmoid(net) , np.subtract( np.ones(net.shape) , sigmoid(net)))


def tanh(net):
    return np.tanh(net)


def d_tanh(net):

    return np.subtract(np.ones(net.shape),np.square(np.tanh(net)))

def relu(net):
    return np.maximum(net, 0)

def d_relu(net):
    rel = [1 if x > 0 else 0 for x in net.flatten()]
    net=np.array(rel).reshape(net.shape)

    return net

def leaky_relu(net):
    rel = [x if x >= 0 else 0.01 * x for x in net.flatten()]
    net=np.array(rel).reshape(net.shape)
    return net

def d_leaky_relu(net):
    rel = [1 if x >= 0 else 0.01 for x in net.flatten()]
    net=np.array(rel).reshape(net.shape)
    return net

activations = {
    C.SIGMOID: sigmoid,
    C.TANH: tanh,
    C.RELU: relu,
    C.LEAKYRELU: leaky_relu,
    C.IDENTITY:identity,
}
derivatives = {
    C.SIGMOID: d_sigmoid,
    C.TANH: d_tanh,
    C.RELU: d_relu,
    C.LEAKYRELU: d_leaky_relu,
    C.IDENTITY:d_identity
}

