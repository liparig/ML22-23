from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV 
import readMonk_and_Cup as readMC
import costants as C
from dnn_plot import plot_curves 
import os

def monk1_evaluation(theta,prefixFilename,fold=2):
    TR_x_monk1,TR_y_monk1 = readMC.get_train_Monk_1()
    TS_x_monk1,TS_y_monk1= readMC.get_test_Monk_1()
    kfCV = KfoldCV(TR_x_monk1, TR_y_monk1, fold) 
    winner = kfCV.validate(inTheta =  theta, FineGS = True, prefixFilename=prefixFilename)
    winnerTheta=winner.get_dictionary()
    errors=holdoutTest(winnerTheta,TR_x_monk1,TR_y_monk1,TS_x_monk1,TS_y_monk1)
    savePlotFig(errors,"TestMonk_1",prefixFilename,f"Test Monk 1 {prefixFilename}",theta=winnerTheta)

def holdoutTest(winner,TR_x_set,TR_y_set,TS_x_set,TS_y_set,):
    # Hold-out Test 1
    model = dnn(**winner)
    errors=model.fit(TR_x_set,TR_y_set,TS_x_set,TS_y_set)
    out = model.forward_propagation(TS_x_set)
    classificationAccuracy = model.metrics.metrics_binary_classification(TS_y_set, out)
    print("Test Accuracy:", classificationAccuracy[C.ACCURACY], "\nClassified:", classificationAccuracy[C.CLASSIFIED], "Missclassified:", classificationAccuracy[C.MISSCLASSIFIED],"Precision",classificationAccuracy[C.PRECISION])
    return errors

def savePlotFig(errors,DirName,FileName,Title,theta):
    # Path
    path_dir_models_coarse = os.path.join(C.PATH_PLOT_DIR,DirName)
    if not os.path.exists(path_dir_models_coarse):
            os.makedirs(path_dir_models_coarse)
    #error_tr=False if errors['error']-errors['loss']==0 else errors['loss']
    plot_curves(errors['error'], errors['validation'], errors['metric_tr'], errors['metric_val'], error_tr=error_tr,
                        lbl_tr = C.LABEL_PLOT_TRAINING, lbl_vs = C.LABEL_PLOT_VALIDATION, path = path_dir_models_coarse+"/"+FileName, 
                        ylim = (-0.5, 1.5), titleplot = Title,
                        theta = theta, labelsY = ['Loss',  C.ACCURACY])
     
def main():
        theta_batch = {C.L_NET:[[17,3,1]],
            C.L_ACTIVATION:[[C.TANH]],
            C.L_ETA:[0.8 ,0.5],
            C.L_TAU: [(500,0.2)],
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
        monk1_evaluation(theta=theta_batch,prefixFilename="_Batch_",fold=2)
        
        theta_mini = theta_batch.copy()
        theta_mini[C.L_ETA]=[0.007,0.0007]
        theta_mini[C.L_TAU]=[(200, 0.0007),(100, 0.00007)]
        theta_mini[C.L_DIMBATCH]=[1,25]
        theta_mini[C.L_MOMENTUM]= [(False,False),(C.NESTEROV,0.8),(C.CLASSIC,0.8)]

        monk1_evaluation(theta=theta_mini,prefixFilename="_Mini_",fold=2)

    
if __name__ == "__main__":
    main()