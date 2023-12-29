from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV
import costants as C
from dnnPlot import plot_curves 
import os
import kfoldLog
import readMonkAndCup as readMC
import numpy as np
import time 

def cup_evaluation(TR_x_cup, TR_y_cup, TS_x_cup, TS_y_cup, theta, dirName, prefixFilename, fold = 2):
    kfCV = KfoldCV(TR_x_cup, TR_y_cup, fold) 
    winner,meanmetrics = kfCV.validate(inTheta =  theta, FineGS = True, prefixFilename = prefixFilename)
    winnerTheta=winner.get_dictionary()
    trerrors,euclidianAccuracy,results=holdoutTest(winnerTheta, TR_x_cup, TR_y_cup, TS_x_cup, TS_y_cup,val_per=0.25,meanepochs=int(meanmetrics['mean_epochs']))
    savePlotFig(trerrors, dirName, prefixFilename, f"{dirName}{prefixFilename}", theta = winnerTheta)
    log,timestamp=kfoldLog.Model_Assessment_log(dirName,prefixFilename,f"Model Hyperparameters:\n {winnerTheta}\n",f"Model Selection Result obtained in {fold}# folds:\n{meanmetrics}\n Mean Euclidian Error:\n{euclidianAccuracy}\n")
    kfoldLog.Model_Assessment_Outputs(results,DIRNAME,prefixFilename,timestamp)


def holdoutTest(winner,TR_x_set,TR_y_set,TS_x_set,TS_y_set,val_per=0.25,meanepochs=0):
    # Hold-out Test 1
    model = dnn(**winner)
    if val_per>0:
        tr_x,tr_y,val_x,val_y=readMC.split_Tr_Val(TR_x_set,TR_y_set,perc=val_per)
        errors=model.fit(tr_x,tr_y,val_x,val_y,TS_x_set,TS_y_set)
        print("Size Dataset x", tr_x.shape,"y",tr_y.shape,"valx",val_x.shape,"valy",val_y.shape)
    else:
        model.epochs=meanepochs
        errors=model.fit(TR_x_set,TR_y_set,[],[],TS_x_set,TS_y_set)
    out = model.forward_propagation(TS_x_set)
    euclidianAccuracy = model.metrics.mean_euclidean_error(TS_y_set, out)
    # Scrivo risultati su file
    # Unisco gli array lungo l'asse delle colonne
    result = np.concatenate((TS_y_set, out), axis=1)
    
    print("Test euclidianError:", euclidianAccuracy)
    
    return errors,euclidianAccuracy,result

def savePlotFig(errors, dirName, fileName, title, theta):
    # Path
    path_dir_models_coarse = os.path.join(C.PATH_PLOT_DIR, dirName)
    if not os.path.exists(path_dir_models_coarse):
            os.makedirs(path_dir_models_coarse)
    # is false if the loss is zero else take the loss 
    inError_tr = False if errors['loss']==0 else errors['loss']
    labelError='validation'
    metric='metric_val'
    if len(errors['test'])>0:
        labelError='test'
        metric='metric_test'
    plot_curves(errors['error'], errors[labelError], errors['metric_tr'], errors[metric], error_tr = inError_tr,
                        lbl_tr = C.LABEL_PLOT_TRAINING, lbl_vs = labelError.capitalize(), path = f"{path_dir_models_coarse}/{fileName}", 
                        ylim = (-0.5, 10),yMSElim=(0,(errors['error'][-1])*3) ,titleplot = title,
                        theta = theta, labelsY = ['Loss',  "MEE"])
     
def main(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, dirName):
    theta_batch = {
        C.L_NET:[[9,20,3],[9,15,5,3]],
        C.L_ACTIVATION:[[C.RELU,C.RELU,C.IDENTITY],[C.TANH,C.IDENTITY]],
        C.L_ETA:[0.003],
        C.L_TAU: [(100,0.01)],
        C.L_REG:[(C.LASSO,0.001),(C.TIKHONOV,0.001)],
        C.L_DIMBATCH:[0],
        C.L_MOMENTUM: [(C.CLASSIC,0.6)],
        C.L_EPOCHS:[500],
        C.L_SHUFFLE:True,
        C.L_EPS: [0.001],
        C.L_DISTRIBUTION:[C.GLOROT],
        C.L_BIAS:[0],
        C.L_SEED: [52],
        C.L_CLASSIFICATION:False,
        C.L_EARLYSTOP:True,
        C.L_PATIENCE: [10],
        C.L_TRESHOLD_VARIANCE:[1.e-2]    
    }
    
    cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, theta_batch, dirName, prefixFilename = C.PREFIXBATCH, fold = 5)
    
    theta_mini = {
        C.L_NET:[[9,20,3],[9,15,5,3]],
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
    }
    
    cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, theta_mini, dirName, prefixFilename = C.PREFIXMINIBATCH, fold = 2)
    
if __name__ == "__main__":
    
    TR_x_CUP, TR_y_CUP,  TS_x_CUP, TS_y_CUP = readMC.get_cup_house_test()
    DIRNAME:str = "TestCUP_1"
        
    main(TR_x_CUP, TR_y_CUP, TS_x_CUP, TS_y_CUP, DIRNAME)