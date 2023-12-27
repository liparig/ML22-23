#!/usr/bin/env python
# coding: utf-8

import numpy as np

class MSE:
    """
        Computes the (M)SE between the target vector Y and the output predicted Y_hat
        return: loss in terms of squared error (divided for 2 to semplify the derivative)
        """
    def loss(self, Y, Y_hat):
        #l = 2 * Y.shape[0]
        dp = np.subtract(np.squeeze(Y), np.squeeze(Y_hat))
        squares = np.square(dp)
        loss = np.sum(squares)
        loss = loss * (1/2)
        return loss

    """
        Computes the derivative of the MSE between the target vector  Y and the output predicted Y_hat
        return derivative of the squared error (2 of the exponent was delete by the denominator constant 2)
    """
    def d_loss(self, Y, Y_hat):
        return np.subtract(np.squeeze(Y), np.squeeze(Y_hat))
        #return np.subtract(np.squeeze(Y), np.squeeze(Y_hat))/len(Y_hat)
        #return (np.squeeze(Y) - np.squeeze(Y_hat))/len(Y_hat)


loss={
    'mse':MSE
}