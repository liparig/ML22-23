from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV 
import readMonkAndCup as readMC
import costants as C
from dnnPlot import plot_curves 
import os

def monk_evaluation(TR_x_monk, TR_y_monk, TS_x_monk, TS_y_monk, theta, dirName, prefixFilename, fold = 2):
    kfCV = KfoldCV(TR_x_monk, TR_y_monk, fold) 
    winner = kfCV.validate(inTheta =  theta, FineGS = True, prefixFilename = prefixFilename)
    winnerTheta=winner.get_dictionary()
    errors=holdoutTest(winnerTheta, TR_x_monk, TR_y_monk, TS_x_monk, TS_y_monk)
    savePlotFig(errors, dirName, prefixFilename, f"{dirName}{prefixFilename}", theta = winnerTheta)

def holdoutTest(winner,TR_x_set,TR_y_set,TS_x_set,TS_y_set,):
    # Hold-out Test 1
    model = dnn(**winner)
    errors=model.fit(TR_x_set,TR_y_set,TS_x_set,TS_y_set)
    out = model.forward_propagation(TS_x_set)
    classificationAccuracy = model.metrics.metrics_binary_classification(TS_y_set, out)
    print("Test Accuracy:", classificationAccuracy[C.ACCURACY], "\nClassified:", classificationAccuracy[C.CLASSIFIED], "Missclassified:", classificationAccuracy[C.MISSCLASSIFIED],"Precision",classificationAccuracy[C.PRECISION])
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
     
def main(inTR_x_monk, inTR_y_monk, inTS_x_monk, inTS_y_monk, dirName):
    theta_batch = {
        C.L_NET:[[17,2,1]], #unit for layer between 2 and 4. it's written in the slide
        C.L_ACTIVATION:[[C.TANH]],
        C.L_ETA:[0.8 ,0.5],
        C.L_TAU: [(500, 0.2)],
        C.L_REG:[(False,False),(C.TIKHONOV,0.01)],
        C.L_DIMBATCH:[0],
        C.L_MOMENTUM: [(False,False)],
        C.L_EPOCHS:[2000],
        C.L_SHUFFLE:True,
        C.L_EPS: [0.2],
        C.L_DISTRIBUTION:[C.UNIFORM],
        C.L_BIAS:[0],
        C.L_SEED: [52],
        C.L_CLASSIFICATION:True,
        C.L_EARLYSTOP:True,
        C.L_PATIENCE: [20,50],
        C.L_TRESHOLD_VARIANCE:[C.TRESHOLDVARIANCE]    
    }
    
    monk_evaluation(inTR_x_monk, inTR_y_monk, inTS_x_monk, inTS_y_monk, theta_batch, dirName, prefixFilename = C.PREFIXBATCH, fold = 2)
    
    theta_mini = theta_batch.copy()
    theta_mini[C.L_ETA]=[0.007,0.0007]
    theta_mini[C.L_TAU]=[(200, 0.0007),(100, 0.00007)]
    theta_mini[C.L_DIMBATCH]=[1,25]
    theta_mini[C.L_MOMENTUM]= [(False,False),(C.NESTEROV,0.8),(C.CLASSIC,0.8)]
    
    monk_evaluation(inTR_x_monk, inTR_y_monk, inTS_x_monk, inTS_y_monk, theta_mini, dirName, prefixFilename = C.PREFIXMINIBATCH, fold = 2)
    
if __name__ == "__main__":
    
    TR_x_monk1, TR_y_monk1 = readMC.get_train_Monk_1()
    TS_x_monk1, TS_y_monk1 = readMC.get_test_Monk_1()
    DIRNAME:str = "TestMonk_1"
    
    # TR_x_monk2, TR_y_monk2 = readMC.get_train_Monk_2()
    # TS_x_monk2, TS_y_monk2 = readMC.get_test_Monk_2()
    # DIRNAME:str = "TestMonk_2"
    
    # TR_x_monk3, TR_y_monk3 = readMC.get_train_Monk_3()
    # TS_x_monk3, TS_y_monk3 = readMC.get_test_Monk_3()
    # DIRNAME:str = "TestMonk_3"
    
    main(TR_x_monk1, TR_y_monk1, TS_x_monk1, TS_y_monk1, DIRNAME)