from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV 
import kfoldLog
import readMonkAndCup as readMC
import costants as C
from dnnPlot import plot_curves 
import os

def monk_evaluation(TR_x_monk, TR_y_monk, TS_x_monk, TS_y_monk, theta, dirName, prefixFilename, fold = 2):
    kfCV = KfoldCV(TR_x_monk, TR_y_monk, fold) 
    winner,meanmetrics = kfCV.validate(inTheta =  theta, FineGS = False, prefixFilename = dirName+prefixFilename)
    winnerTheta=winner.get_dictionary()
    trerrors,classification=holdoutTest(winnerTheta, TR_x_monk, TR_y_monk, TS_x_monk, TS_y_monk,val_per=0,meanepochs=int(meanmetrics['mean_epochs']))
    savePlotFig(trerrors, dirName, prefixFilename, f"{dirName}{prefixFilename}", theta = winnerTheta)
    kfoldLog.Model_Assessment_log(dirName,prefixFilename,f"Model Hyperparameters:\n {str(winnerTheta)}\n",f"Model Selection Result obtained in {fold}# folds:\n{meanmetrics}\nClassification values in test:\n{classification}\n Errors in re-trainings:\n{trerrors['epochs']}\n")

def holdoutTest(winner,TR_x_set,TR_y_set,TS_x_set,TS_y_set,val_per=0.25,meanepochs=0):
    # Hold-out Test 1
    model = dnn(**winner)
    if val_per>0:
        tr_x,tr_y,val_x,val_y=readMC.split_Tr_Val(TR_x_set,TR_y_set,perc=val_per)
        errors=model.fit(tr_x,tr_y,val_x,val_y,TS_x_set,TS_y_set)
        print("Size Dataset x", tr_x.shape,"y",tr_y.shape,"valx",val_x.shape,"valy",val_y.shape)
    else:
        model.early_stop=False
        model.epochs=meanepochs
        errors=model.fit(TR_x_set,TR_y_set,[],[],TS_x_set,TS_y_set)
   
    
    out = model.forward_propagation(TS_x_set)
    classificationAccuracy = model.metrics.metrics_binary_classification(TS_y_set, out)
    print("Test Accuracy:", classificationAccuracy[C.ACCURACY], "\nClassified:", classificationAccuracy[C.CLASSIFIED], "Missclassified:", classificationAccuracy[C.MISSCLASSIFIED],"Precision",classificationAccuracy[C.PRECISION])
    
    return errors,classificationAccuracy

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
                        ylim = (-0.5, 1.5), titleplot = title,
                        theta = theta, labelsY = ['Loss',  C.ACCURACY])
     
def main(inTR_x_monk, inTR_y_monk, inTS_x_monk, inTS_y_monk, dirName):
    theta_batch = {
        C.L_NET:[[17,2,1]], #unit for layer between 2 and 4. it's written in the slide
        C.L_ACTIVATION:[[C.LEAKYRELU,C.TANH]],
        C.L_ETA:[0.9],
        C.L_TAU: [(False,False)],
        C.L_REG:[(False,False)],
        C.L_DIMBATCH:[0],
        C.L_MOMENTUM: [(False,False)],
        C.L_EPOCHS:[500],
        C.L_SHUFFLE:True,
        C.L_EPS: [0.01],
        C.L_DISTRIBUTION:[C.UNIFORM],
        C.L_BIAS:[0],
        C.L_SEED: 25,
        C.L_CLASSIFICATION:True,
        C.L_EARLYSTOP:True,
        C.L_PATIENCE: [20],
        C.L_TRESHOLD_VARIANCE:[1.e-10]    
    }
    
    #monk_evaluation(inTR_x_monk, inTR_y_monk, inTS_x_monk, inTS_y_monk, theta_batch, dirName, prefixFilename = C.PREFIXBATCH, fold = 2)
    
    theta_mini = theta_batch.copy()
    theta_mini[C.L_ETA]=[0.09, 0.03]
    theta_mini[C.L_TAU]=[(100, 0.003)]
    theta_mini[C.L_DIMBATCH]=[25]
    
    #monk_evaluation(inTR_x_monk, inTR_y_monk, inTS_x_monk, inTS_y_monk, theta_mini, dirName, prefixFilename = C.PREFIXMINIBATCH, fold = 5)
    
    theta_batch_NOES = theta_batch.copy()
    theta_batch_NOES[C.L_EARLYSTOP]= False
    theta_batch_NOES[C.L_EPOCHS]=[500]
    
    monk_evaluation(inTR_x_monk, inTR_y_monk, inTS_x_monk, inTS_y_monk, theta_batch_NOES, dirName, prefixFilename = C.PREFIXBATCH+"NO_ES", fold = 2)
    
    theta_mini_NoES = theta_mini.copy()
    theta_mini_NoES[C.L_EARLYSTOP]=False
    theta_mini_NoES[C.L_EPOCHS]=[210,500]


    
    #monk_evaluation(inTR_x_monk, inTR_y_monk, inTS_x_monk, inTS_y_monk, theta_mini_NoES, dirName, prefixFilename = C.PREFIXMINIBATCH+"NO_ES", fold = 5)
    
    
if __name__ == "__main__":
    
   # TR_x_monk1, TR_y_monk1 = readMC.get_train_Monk_1()
    #TS_x_monk1, TS_y_monk1 = readMC.get_test_Monk_1()
    #DIRNAME:str = "TestMonk_1"
    
    TR_x_monk2, TR_y_monk2 = readMC.get_train_Monk_2()
    TS_x_monk2, TS_y_monk2 = readMC.get_test_Monk_2()
    DIRNAME:str = "TestMonk_2"
    """
    TR_x_monk3, TR_y_monk3 = readMC.get_train_Monk_3()
    TS_x_monk3, TS_y_monk3 = readMC.get_test_Monk_3()
    DIRNAME:str = "TestMonk_3"
    """  
    main(TR_x_monk2, TR_y_monk2, TS_x_monk2, TS_y_monk2, DIRNAME)