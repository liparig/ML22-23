import numpy as np

class tikhonov_regularization:
    """
        Computes Tikhonov regularization on the weitghs
        :param w: weights of one layer
        :param tlambda: regularization coefficient
        """
    def penalty(self, tlambda, w):
        return tlambda * np.square(np.linalg.norm(w))

    """
    Computes the derivative of Tikhonov regularization
    :param w: weights of one layer
    :param tlambda:  regularization coefficient
    """
    def derivative(self, tlambda, w):
        return 2 * tlambda * w       

class lasso_regularization:
    """
        Computes Lasso regularization on the  weights
        :param w: weights of one layer
        :param llambda: regularization coefficient
        """
    def penalty(self, llambda, w):
        return llambda * np.sum(np.abs(w))

    """
    Computes the derivative of the Lasso regularization (L1)
    :param w: weights of one layer
    :param llambda:  regularization coefficient
    """
    def derivative(self, llambda, w):
        lasso=[]
        for j in w.flatten():
                if j < 0:
                    lasso.append(-llambda)
                elif j > 0:
                    lasso.append(llambda)
                else:
                    lasso.append(0)
        lasso= np.array(lasso)
        lasso=lasso.reshape(w.shape)
        return lasso


regularization = {
    'tikhonov': tikhonov_regularization,
    'lasso': lasso_regularization
}