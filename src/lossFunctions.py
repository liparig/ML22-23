#!/usr/bin/env python
# coding: utf-8

import numpy as np
class MSE:
    """
        Computes the MSE between the target vector Y and the output predicted Y_hat
        return: loss in terms of squared error (divided for 2 to semplify the derivative)
        """
    def loss(Y, Y_hat):
        l=2*Y.shape[0]
        dp = np.squeeze(Y) - np.squeeze(Y_hat)
        squares=np.square(dp)
        loss=np.sum(squares)
        loss =  loss * (1/l)
        return loss

    """
        Computes the derivative of the MSE between the target vector  Y and the output predicted Y_hat
        return derivative of the squared error (2 of the exponent was delete by the denominator constant 2)
    """
    def d_loss(Y, Y_hat):
        return (np.squeeze(Y) - np.squeeze(Y_hat)) * -1

loss={
    'mse':MSE
}