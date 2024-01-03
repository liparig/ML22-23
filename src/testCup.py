from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV
import costants as C
from dnnPlot import plot_curves 
import os
import kfoldLog
import readMonkAndCup as readMC
import numpy as np

#it's a test for get the performance with test dataset

# Evaluete the cup dataset
# :param: TR_x_cup is training dataset
# :param: TR_y_cup is training targets dataset
# :param: TS_x_cup is test dataset
# :param: TS_y_cup is test targets dataset
# :param: theta is configuration object
# :param: dirName is directory name
# :param: prefixFilename is prefix of the file name
# :param: fold is number of the folds
def cup_evaluation(TR_x_cup, TR_y_cup, TS_x_cup, TS_y_cup, theta, dirName, prefixFilename, fold = 2):
    #MODEL SELECTION KFCV
    kfCV = KfoldCV(TR_x_cup, TR_y_cup, fold)
    winner, meanmetrics = kfCV.validate(inTheta =  theta, FineGS = True, prefixFilename = prefixFilename)
    winnerTheta = winner.get_dictionary()

    #MODEL ASSESSMENT HOLDOUT
    trerrors, euclidianAccuracy, results = holdoutTest(winnerTheta, TR_x_cup, TR_y_cup, TS_x_cup, TS_y_cup, val_per = 0.25, meanepochs = int(meanmetrics['mean_epochs']))
    _, timestamp = kfoldLog.Model_Assessment_log(dirName, prefixFilename, f"Model Hyperparameters:\n {winnerTheta}\n", f"Model Selection Result obtained in {fold}# folds:\n{meanmetrics}\n Mean Euclidian Error:\n{euclidianAccuracy}\n")
    kfoldLog.Model_Assessment_Outputs(
        results, dirName, f'{prefixFilename}_HouseTest', timestamp
    )
    savePlotFig(trerrors, dirName, prefixFilename, f"{dirName}{prefixFilename}", theta = winnerTheta)

    #BLIND TEST
    TS_x_CUP_blind = readMC.get_blind_test_CUP()
    TR_x_CUP_All, TR_y_CUP_All,_,_ = readMC.get_cup_house_test(perc = 0)
    _, _, resultsBlind = holdoutTest(winnerTheta,  TR_x_CUP_All, TR_y_CUP_All, TS_x_CUP_blind, [], val_per =  0.25, meanepochs = int(meanmetrics['mean_epochs']))
    kfoldLog.ML_Cup_Template(resultsBlind, dirName, f'{prefixFilename}_blind', timestamp)
    return winnerTheta

# Execute the holdout test
# :param: winner is the configuration object
# :param: TR_x_cup is training dataset
# :param: TR_y_cup is training targets dataset
# :param: TS_x_cup is test dataset
# :param: TS_y_cup is test targets dataset
# :param: val_per is the percent of the data for testing
# :param: meanepochs for early stop and fitting epoch
# :return: errors object, the accuracy property and the result
def holdoutTest(winner, TR_x_set, TR_y_set, TS_x_set, TS_y_set, val_per:float = 0.25, meanepochs:int = 0):
    # Hold-out Test 1
    model = dnn(**winner)
    #Training model with or witout validation
    if val_per > 0:
        tr_x, tr_y, val_x, val_y = readMC.split_Tr_Val(TR_x_set, TR_y_set, perc = val_per)
        errors = model.fit(tr_x, tr_y, val_x, val_y, TS_x_set, TS_y_set)
        print("Size Dataset x", tr_x.shape, "y", tr_y.shape, "valx", val_x.shape, "valy", val_y.shape)
    else:
        model.epochs = meanepochs
        errors = model.fit(TR_x_set, TR_y_set, [], [], TS_x_set, TS_y_set)

    #MAKES PREDICTION
    out = model.forward_propagation(TS_x_set)

    #CHECK IF BLIND
    if(not isinstance(TS_y_set, list)):
        euclidianAccuracy = model.metrics.mean_euclidean_error(TS_y_set, out)
        result = np.concatenate((TS_y_set, out), axis=1)
        print("Test euclidianError:", euclidianAccuracy)
    else:
        result = out
        euclidianAccuracy = None
    
    return errors, euclidianAccuracy, result

def ensemble_Cup(models,tr_x, tr_y,Ts_x,Ts_y, dirname, filename):
    inputs = readMC.get_blind_test_CUP() if (isinstance(Ts_x, list)) else Ts_x
    outs = []
    errors = []
    for model in models:
        model= dnn(**model)
        errors.append(model.fit(tr_x, tr_y))
        outs.append(model.forward_propagation(inputs))
    mean_outs = np.mean(outs, axis=0)
    #CHECK IF BLIND
    if (not isinstance(Ts_y, list)):
        euclidianAccuracy = model.metrics.mean_euclidean_error(Ts_y, mean_outs)
        result = np.concatenate((Ts_y, mean_outs), axis=1)
        print("Ensembe EuclidianError:", euclidianAccuracy)
        kfoldLog.Model_Assessment_Outputs(result, dirname, filename)
    else:
        result = mean_outs
        euclidianAccuracy = None
        kfoldLog.ML_Cup_Template(result, dirname, f'{filename}_Blind')
   

def savePlotFig(errors, dirName, fileName, title, theta):
    # Path
    path_dir_models_coarse = os.path.join(C.PATH_PLOT_DIR, dirName)
    if not os.path.exists(path_dir_models_coarse):
            os.makedirs(path_dir_models_coarse)
    # is false if the loss is zero else take the loss 
    inError_tr = False if errors['loss']==0 else errors['loss']
    labelError = 'validation'
    metric = 'metric_val'
    if len(errors['test']) > 0:
        labelError = 'test'
        metric = 'metric_test'
    plot_curves(errors['error'], errors[labelError], errors['metric_tr'], errors[metric], error_tr = inError_tr,
                        lbl_tr = C.LABEL_PLOT_TRAINING, lbl_vs = labelError.capitalize(), path = f"{path_dir_models_coarse}/{fileName}", 
                        ylim = (-0.5, 10),yMSElim=(0,(errors['error'][-1])*100) ,titlePlot = title,
                        theta = theta, labelsY = ['Loss',  "MEE"])
     
def main(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, dirName):
    """theta_batch = {
        C.L_NET:[[10,15,5,3]],
        C.L_ACTIVATION:[[C.RELU,C.RELU,C.IDENTITY],[C.TANH,C.IDENTITY],[C.LEAKYRELU,C.IDENTITY],[C.LEAKYRELU,C.LEAKYRELU,C.IDENTITY],[C.SIGMOID,C.TANH,C.IDENTITY]],
        C.L_ETA:[0.03, 0.1],
        C.L_TAU: [(200,0.003), (False,False)],
        C.L_REG:[(C.LASSO,0.01),(C.TIKHONOV,0.01), (False,False)],
        C.L_DIMBATCH:[0],
        C.L_MOMENTUM: [(C.CLASSIC,0.9), (False,False)],
        C.L_EPOCHS:[500],
        C.L_SHUFFLE:True,
        C.L_EPS: [0.01],
        C.L_DISTRIBUTION:[C.GLOROT],
        C.L_G_CLIPPING:[(True,5)],
        C.L_BIAS:[0],
        C.L_SEED: [52],
        C.L_CLASSIFICATION:False,
        C.L_EARLYSTOP:True,
        C.L_PATIENCE: [10],
        C.L_TRESHOLD_VARIANCE:[1.e-2]    
    }"""
    theta_batch = {
        C.L_NET:[[10,8,5,3],[10,15,3]],
        C.L_ACTIVATION:[[C.TANH,C.IDENTITY],[C.LEAKYRELU,C.IDENTITY],[C.LEAKYRELU,C.LEAKYRELU,C.IDENTITY],[C.TANH,C.TANH,C.IDENTITY]],
        C.L_ETA:[0.004,0.008],
        C.L_TAU: [(False,False),(500,0.0008)],
        C.L_REG:[(C.TIKHONOV,0.002),(C.LASSO,0.002),(False,False)],
        C.L_DIMBATCH:[0],
        C.L_MOMENTUM: [(C.CLASSIC, 0.7),(C.NESTEROV, 0.7)],
        C.L_EPOCHS:[2000],
        C.L_SHUFFLE:True,
        C.L_EPS: [0.01],
        C.L_DISTRIBUTION:[C.GLOROT],
        C.L_G_CLIPPING:[(True,7),C.G_CLIPPING],
        C.L_DROPOUT:[C.DROPOUT],
        C.L_BIAS:[0],
        C.L_SEED: [52],
        C.L_CLASSIFICATION:False,
        C.L_EARLYSTOP:True,
        C.L_PATIENCE: [10,20],
        C.L_TRESHOLD_VARIANCE:[1.e-10]    
    }

    batchWinner=cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, theta_batch, dirName, prefixFilename = C.PREFIXBATCH, fold = 5)
    models = [batchWinner]
    """theta_mini = {
        C.L_NET:[[10,20,3],[10,15,5,3]],
        C.L_ACTIVATION:[[C.RELU,C.RELU,C.IDENTITY],[C.TANH,C.IDENTITY]],
        C.L_ETA:[0.00003],
        C.L_TAU: [(20,0.0003)],
        C.L_REG:[(C.LASSO,0.00001),(C.TIKHONOV,0.00001)],
        C.L_DIMBATCH:[1,50,100],
        C.L_MOMENTUM: [(C.CLASSIC,0.6),(C.NESTEROV,0.6)],
        C.L_EPOCHS:[500],
        C.L_SHUFFLE:True,
        C.L_EPS: [0.001],
        C.L_DISTRIBUTION:[C.GLOROT],
        C.L_BIAS:[0],
        C.L_SEED: [52],
        C.L_CLASSIFICATION:False,
        C.L_EARLYSTOP:True,
        C.L_PATIENCE: [10],
        C.L_TRESHOLD_VARIANCE:[1.e-1]    
    }"""
    theta_mini = {
        C.L_NET:[[10,20,15,3],[10,10,8,3]],
        C.L_ACTIVATION:[[C.LEAKYRELU,C.LEAKYRELU,C.IDENTITY],[C.TANH,C.TANH,C.IDENTITY]],
        C.L_ETA:[0.0001],
        C.L_TAU: [(100,0.00005)],
        C.L_REG:[(C.TIKHONOV,0.0001)],
        C.L_DIMBATCH:[100],
        C.L_MOMENTUM: [(C.CLASSIC,0.8)],
        C.L_EPOCHS:[1000],
        C.L_SHUFFLE:True,
        C.L_EPS: [0.001],
        C.L_DISTRIBUTION:[C.GLOROT],
        C.L_G_CLIPPING:[(True,5)],
        C.L_DROPOUT:[(True,0.5)],
        C.L_BIAS:[0],
        C.L_SEED: [52],
        C.L_CLASSIFICATION:False,
        C.L_EARLYSTOP:False,
        C.L_PATIENCE: [10],
        C.L_TRESHOLD_VARIANCE:[1.e-8]    
    }

    minibatchWinner=cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, theta_mini, dirName, prefixFilename = C.PREFIXMINIBATCH, fold = 5)
    models.append(minibatchWinner)
    ensemble_Cup(models,inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup,dirName,filename="Ensemle_")
    
if __name__ == "__main__":
    
    TR_x_CUP, TR_y_CUP,  TS_x_CUP, TS_y_CUP = readMC.get_cup_house_test()
    DIRNAME:str = "TestCUP_1"
    # print(TR_x_CUP.shape, TR_y_CUP.shape,  TS_x_CUP.shape, TS_y_CUP.shape)
    # print(TR_x_CUP[0], TR_y_CUP[0],  TS_x_CUP[0], TS_y_CUP[0])
    # input('premi')
    main(TR_x_CUP, TR_y_CUP, TS_x_CUP, TS_y_CUP, DIRNAME)