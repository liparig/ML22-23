from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV
import costants as C
from dnnPlot import plot_curves 
import os

import readMonkAndCup as readMC

def cup_evaluation(TR_x_cup, TR_y_cup, TS_x_cup, TS_y_cup, theta, dirName, prefixFilename, fold = 2):
    kfCV = KfoldCV(TR_x_cup, TR_y_cup, fold) 
    winner = kfCV.validate(inTheta =  theta, FineGS = True, prefixFilename = prefixFilename)
    winnerTheta=winner.get_dictionary()
    errors=holdoutTest(winnerTheta, TR_x_cup, TR_y_cup, TS_x_cup, TS_y_cup)
    savePlotFig(errors, dirName, prefixFilename, f"{dirName}{prefixFilename}", theta = winnerTheta)

def holdoutTest(winner,TR_x_set,TR_y_set,TS_x_set,TS_y_set,):
    # Hold-out Test 1
    model = dnn(**winner)
    errors=model.fit(TR_x_set,TR_y_set,TS_x_set,TS_y_set)
    out = model.forward_propagation(TS_x_set)
    euclidianAccuracy = model.metrics.mean_euclidean_error(TS_y_set, out)
    print("Test euclidianError:", euclidianAccuracy)
    return errors

def savePlotFig(errors, dirName, fileName, title, theta):
    # Path
    path_dir_models_coarse = os.path.join(C.PATH_PLOT_DIR, dirName)
    if not os.path.exists(path_dir_models_coarse):
            os.makedirs(path_dir_models_coarse)
    # error_tr=False if errors['error']-errors['loss']==0 else errors['loss']
    plot_curves(errors['error'], errors['validation'], errors['metric_tr'], errors['metric_val'], error_tr=False,
                        lbl_tr = C.LABEL_PLOT_TRAINING, lbl_vs = C.LABEL_PLOT_VALIDATION, path = f"{path_dir_models_coarse}/{fileName}", 
                        ylim = (-0.5, 1.5), titleplot = title,
                        theta = theta, labelsY = ['Loss',  C.ACCURACY])
     
def main(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, dirName):
    theta_batch = {
        C.L_NET:[[9,16,16,3]],
        C.L_ACTIVATION:[[C.RELU,C.RELU,C.IDENTITY]],
        C.L_ETA:[0.5, 0.8],
        C.L_TAU: [(False,False), (1000,0.005)],
        C.L_REG:[(C.TIKHONOV,0.5), (C.LASSO,0.5)],
        C.L_DIMBATCH:[0],
        C.L_MOMENTUM: [(False,False)],
        C.L_EPOCHS:[500, 1000],
        C.L_SHUFFLE:True,
        C.L_EPS: [0.1],
        C.L_DISTRIBUTION:[C.UNIFORM],
        C.L_BIAS:[0],
        C.L_SEED: [52],
        C.L_CLASSIFICATION:False,
        C.L_EARLYSTOP:True,
        C.L_PATIENCE: [400],
        C.L_TRESHOLD_VARIANCE:[C.TRESHOLDVARIANCE]    
    }
    
    cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, theta_batch, dirName, prefixFilename = C.PREFIXBATCH, fold = 5)
    
    # theta_mini = theta_batch.copy()
    # theta_mini[C.L_ETA]=[0.007,0.0007]
    # theta_mini[C.L_TAU]=[(200, 0.0007),(100, 0.00007)]
    # theta_mini[C.L_DIMBATCH]=[1,50,100]
    # theta_mini[C.L_MOMENTUM]= [(False,False),(C.NESTEROV,0.8),(C.CLASSIC,0.8)]
    
    # cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, theta_mini, dirName, prefixFilename = C.PREFIXMINIBATCH, fold = 2)
    
if __name__ == "__main__":
    
    TR_x_CUP, TR_y_CUP,  TS_x_CUP, TS_y_CUP = readMC.get_cup_house_test()
    DIRNAME:str = "TestCUP_1"
        
    main(TR_x_CUP, TR_y_CUP, TS_x_CUP, TS_y_CUP, DIRNAME)