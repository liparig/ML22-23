import numpy as np

class DNN_metrics:
    '''
    Compute metrics_binary_classification
    :param y: target values
    :param y_hat: predicted values
    :param treshold: treshold values
    :return result: dictionary with: 'accuracy','precision','recall','specificity','balanced'
    '''
    def metrics_binary_classification(self,y, y_hat,treshold=0.5):
        if np.squeeze(y).shape != np.squeeze(y_hat).shape:
            raise Exception(f"Sets have different shape Y:{y.shape} Y_hat:{y_hat.shape}")

        tp,tn,fp,fn=0,0,0,0
        for predicted, target in zip(y_hat.flatten(),y.flatten()):
            if predicted < treshold:
                if target == 0:
                    tn+=1
                else:
                    fn+=1
            else:
                if target ==1:
                    tp+=1
                else:
                    fp+=1

        accuracy=(tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn >0 else 0
        recall= tp/(tp+fn) if tp+fn >0 else 0
        precision = tp/(tp+fp) if tp+fp >0 else 0
        specificity= tn/(tn+fp) if tn+fp >0 else 0
        balanced=0.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp  >0 else 0

        return {'misclassified': fp+fn,
        'classified':tp+tn,
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'specificity':specificity,
        'balanced':balanced
        } 

    '''
    Compute mean_absolute_error
    :param y: target values
    :param y_hat: predicted values
    :return mae: mean absolute error
    '''
    def mean_absolute_error(self,y, y_hat):
        sum_error = 0.0
        for predicted, target in zip(y_hat.flatten(),y.flatten()):
            sum_error += abs(predicted - target)
        
        return sum_error/len(y.flatten())

    '''
    Compute root mean squared error
    :param y: target values
    :param y_hat: predicted values
    :return rmse: root mean squared error value
    '''
    def root_mean_squared_error(self,y, y_hat):
        sum_error = 0.0
        for predicted, target in zip(y_hat.flatten(),y.flatten()):
            prediction_error = predicted - target
            sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(y))
        return np.sqrt(mean_error)
    '''
    Compute Mean Euclidean Error
    :param y: target values
    :param y_hat: predicted values
    :return MEE: root mean squared error value
    '''
    def mean_euclidean_error(self,y, y_hat):
        return np.divide(np.sqrt(np.sum(np.square(np.subtract(y, y_hat)))),float(len(y)))
