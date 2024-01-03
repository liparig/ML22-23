from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV 
import kfoldLog
import readMonkAndCup as readMC
import costants as C
from dnnPlot import draw_async 
from candidateHyperparameters import Candidate
import os
import numpy as np

def monk_KfoldCV_evaluation(TR_x_monk, TR_y_monk, TS_x_monk, TS_y_monk, theta, dirName, prefixFilename, fold = 2):
    kfCV = KfoldCV(TR_x_monk, TR_y_monk, fold) 
    winner,meanmetrics = kfCV.validate(inTheta =  theta, FineGS = True, prefixFilename = dirName+prefixFilename)
    winnerTheta=winner.get_dictionary()
    trerrors,classification,result=holdoutTest(winnerTheta, TR_x_monk, TR_y_monk, TS_x_monk, TS_y_monk, val_per = 0, meanepochs = int(meanmetrics['mean_epochs']))
    savePlotFig(trerrors, dirName, prefixFilename, f"Test_s{dirName}{prefixFilename}", theta = winnerTheta)
    mafile, timestr=kfoldLog.Model_Assessment_log(dirName,prefixFilename,f"Model Hyperparameters:\n {str(winnerTheta)}\n",f"Model Selection Result obtained in {fold}# folds:\n{meanmetrics}\nClassification values in test:\n{classification}\n Errors in re-trainings:\n{trerrors['epochs']}\n")
    kfoldLog.Model_Assessment_Outputs(result,dirName,dirName+prefixFilename,col_names=["Target Class", "Predicted Class"],timestamp=timestr)

def monk_model_evaluation(TR_x_monk, TR_y_monk, TS_x_monk, TS_y_monk, theta, dirName, prefixFilename):
    model=Candidate(theta)
    trerrors,classification, result=holdoutTest(model.get_dictionary(), TR_x_monk, TR_y_monk, TS_x_monk, TS_y_monk, val_per=0, meanepochs = theta[C.L_EPOCHS])
    savePlotFig(trerrors, dirName, prefixFilename, f"{dirName}{prefixFilename}", theta = model.get_dictionary())
    mafile, timestr=kfoldLog.Model_Assessment_log(dirName,prefixFilename,f"Model Hyperparameters:\n {str(model)}\n",f"{trerrors}")
    kfoldLog.Model_Assessment_Outputs(result,dirName,prefixFilename,col_names=["Target Class", "Predicted Class"],timestamp=timestr)

    print("Classification", classification)

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
    result = np.concatenate((TS_y_set, out), axis=1)
    if model.a_functions[-1]==C.TANH:
            classi=(-1,1)
            threshold_accuracy=0.3
    else:
        classi=(0,1)
        threshold_accuracy=0.5
    classificationAccuracy = model.metrics.metrics_binary_classification(TS_y_set, out, treshold= threshold_accuracy,classi = classi)
    print("Test Accuracy:", classificationAccuracy[C.ACCURACY], "\nClassified:", classificationAccuracy[C.CLASSIFIED], "Missclassified:", classificationAccuracy[C.MISSCLASSIFIED],"Precision",classificationAccuracy[C.PRECISION])
    
    return errors,classificationAccuracy,result

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
    draw_async(errors['error'], errors[labelError], errors['metric_tr'], errors[metric], error_tr = inError_tr,
                        lbl_tr = C.LABEL_PLOT_TRAINING, lbl_vs = labelError.capitalize(), path = f"{path_dir_models_coarse}/{fileName}", 
                        ylim = (-0.5, 1.5), titlePlot = title,
                        theta = theta, labelsY = ['Loss',  C.ACCURACY])
     
def main():
    theta_batch = {
        C.L_NET:[[17,3,1]], #unit for layer between 2 and 4. it's written in the slide
        C.L_ACTIVATION:[[C.LEAKYRELU,C.SIGMOID],[C.TANH,C.SIGMOID]],
        C.L_ETA:[0.4, 0.8],
        C.L_TAU: [(False,False)],
        C.L_REG:[(C.TIKHONOV,0.001)],
        C.L_G_CLIPPING:[C.G_CLIPPING],
        C.L_DROPOUT:[C.DROPOUT],
        C.L_DIMBATCH:[0],
        C.L_MOMENTUM: [(C.CLASSIC,0.8),(C.NESTEROV, 0.4)],
        C.L_EPOCHS:[1000],
        C.L_SHUFFLE:True,
        C.L_EPS: [0.1],
        C.L_DISTRIBUTION:[C.GLOROT],
        C.L_BIAS:[0],
        C.L_SEED: 25,
        C.L_CLASSIFICATION:True,
        C.L_EARLYSTOP:True,
        C.L_PATIENCE: [10],
        C.L_TRESHOLD_VARIANCE:[1.e-10,1.e-8]    
    }
    for i in range(1,4):
        readTrain = getattr(readMC, f"get_train_Monk_{i}")
        readTest = getattr(readMC, f"get_test_Monk_{i}")
        TR_x_monk, TR_y_monk = readTrain()
        TS_x_monk, TS_y_monk = readTest()
        monk_KfoldCV_evaluation(TR_x_monk, TR_y_monk, TS_x_monk, TS_y_monk, theta_batch, f"KfCV_Monk_{i}", prefixFilename = C.PREFIXBATCH, fold = 3)

        theta_mini = theta_batch.copy()
        theta_mini[C.L_ETA]=[0.009, 0.03]
        theta_mini[C.L_TAU]=[(100, 0.0009)]
        theta_mini[C.L_DIMBATCH]=[25]
        
        monk_KfoldCV_evaluation(TR_x_monk, TR_y_monk, TS_x_monk, TS_y_monk, theta_mini, f"KfCV_Monk_{i}", prefixFilename = C.PREFIXMINIBATCH, fold = 3)

    theta_batch_NOES = theta_batch.copy()
    theta_batch_NOES[C.L_EARLYSTOP]= False
    theta_batch_NOES[C.L_EPOCHS]=[500]

    single_model = {
        C.L_NET:[17,3,1], #unit for layer between 2 and 4. it's written in the slide
        C.L_ACTIVATION:[C.TANH,C.SIGMOID],
        C.L_ETA:[0.8],
        C.L_TAU: (False,False),
        C.L_REG:(False,False),
        C.L_G_CLIPPING:C.G_CLIPPING,
        C.L_DROPOUT:C.DROPOUT,
        C.L_DIMBATCH:0,
        C.L_MOMENTUM: (C.CLASSIC,0.8),
        C.L_EPOCHS:500,
        C.L_SHUFFLE:True,
        C.L_EPS: 0.1,
        C.L_DISTRIBUTION:C.UNIFORM,
        C.L_BIAS:0,
        C.L_SEED: 30,
        C.L_CLASSIFICATION:True,
        C.L_EARLYSTOP:False,
        C.L_PATIENCE:10,
        C.L_TRESHOLD_VARIANCE:1.e-8  
    }
    print("Inizio TestMonk 1:")
    dirName="TestMonk_1"
    theta_mini = single_model.copy()
    theta_mini[C.L_ETA]= 0.08
    theta_mini[C.L_TAU]= C.TAU
    theta_mini[C.L_DIMBATCH]= 15
    theta_mini[C.L_MOMENTUM]= (C.CLASSIC, 0.8)


    TR_x_monk1, TR_y_monk1 = readMC.get_train_Monk_1(single_model[C.L_ACTIVATION][-1]==C.TANH)
    TS_x_monk1, TS_y_monk1 = readMC.get_test_Monk_1(single_model[C.L_ACTIVATION][-1]==C.TANH)
    monk_model_evaluation(TR_x_monk1, TR_y_monk1, TS_x_monk1, TS_y_monk1, single_model, dirName, prefixFilename = C.PREFIXBATCH)
    print("MiniBatch")
    monk_model_evaluation(TR_x_monk1, TR_y_monk1, TS_x_monk1, TS_y_monk1, theta_mini, dirName, prefixFilename = C.PREFIXMINIBATCH)
    print("Fine TestMonk 1:\n")

    print("Inizio TestMonk 2:")
    dirName="TestMonk_2"
    theta_mini[C.L_ETA]= 0.8
    theta_mini[C.L_TAU]= C.TAU
    theta_mini[C.L_MOMENTUM]= (C.NESTEROV, 0.7)
    TR_x_monk2, TR_y_monk2 = readMC.get_train_Monk_2(single_model[C.L_ACTIVATION][-1]==C.TANH)
    TS_x_monk2, TS_y_monk2 = readMC.get_test_Monk_2(single_model[C.L_ACTIVATION][-1]==C.TANH)
    monk_model_evaluation(TR_x_monk2, TR_y_monk2, TS_x_monk2, TS_y_monk2, single_model, dirName, prefixFilename = C.PREFIXBATCH)
    print("MiniBatch")
    monk_model_evaluation(TR_x_monk2, TR_y_monk2, TS_x_monk2, TS_y_monk2, theta_mini, dirName, prefixFilename = C.PREFIXMINIBATCH)

    print("Fine TestMonk 2:\n")

    print("Inizio TestMonk 3:\n")
    dirName="TestMonk_3"
    theta_mini[C.L_MOMENTUM]= (C.CLASSIC, 0.8)
    theta_mini[C.L_ETA]= 0.08
    theta_mini[C.L_TAU]= (100,0.008)
    TR_x_monk3, TR_y_monk3 = readMC.get_train_Monk_3()
    TS_x_monk3, TS_y_monk3 = readMC.get_test_Monk_3()
    monk_model_evaluation(TR_x_monk3, TR_y_monk3, TS_x_monk3, TS_y_monk3, single_model, dirName, prefixFilename = C.PREFIXBATCH)

    print("MiniBatch")
    monk_model_evaluation(TR_x_monk3, TR_y_monk3, TS_x_monk3, TS_y_monk3, theta_mini, dirName, prefixFilename = C.PREFIXMINIBATCH)

    print("Inizio TestMonk 3 LASSO:")
    single_model[C.L_REG]=(C.LASSO,0.0008)
    monk_model_evaluation(
        TR_x_monk3,
        TR_y_monk3,
        TS_x_monk3,
        TS_y_monk3,
        single_model,
        dirName,
        prefixFilename=f"{C.PREFIXBATCH}REGLASSO",
    )

    print("Inizio TestMonk 3 TIKHONOV:")
    single_model[C.L_REG]=(C.TIKHONOV,0.0008)
    monk_model_evaluation(
        TR_x_monk3,
        TR_y_monk3,
        TS_x_monk3,
        TS_y_monk3,
        single_model,
        dirName,
        prefixFilename=f"{C.PREFIXBATCH}REGTIKHONOV",
    )

    print("Inizio TestMonk 3 DropOUT:")
    theta_mini[C.L_NET]=[17,20,1]
    theta_mini[C.L_DIMBATCH]= 15
    theta_mini[C.DROPOUT]=(C.DROPOUT,0.7)
    monk_model_evaluation(
        TR_x_monk3,
        TR_y_monk3,
        TS_x_monk3,
        TS_y_monk3,
        theta_mini,
        dirName,
        prefixFilename=f"{C.PREFIXBATCH}REGDROP",
    )

    print("Fine TestMonk 3:\n")

    
"""def main():
    theta_batch = {
        C.L_NET:[[17,2,1]], #unit for layer between 2 and 4. it's written in the slide
        C.L_ACTIVATION:[[C.TANH]],
        C.L_ETA:[0.8],
        C.L_TAU: [(False,False)],
        C.L_REG:[(False,False)],
        C.L_DIMBATCH:[0],
        C.L_MOMENTUM: [(C.CLASSIC,0.8)],
        C.L_EPOCHS:[500],
        C.L_SHUFFLE:True,
        C.L_EPS: [0.7],
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
    
    single_model = {
        C.L_NET:[17,3,1], #unit for layer between 2 and 4. it's written in the slide
        C.L_ACTIVATION:[C.TANH,C.SIGMOID],
        C.L_ETA:[0.7],
        C.L_TAU: (False,False),
        C.L_REG:(False,False),
        C.L_DIMBATCH:0,
        C.L_MOMENTUM: (C.CLASSIC,0.8),
        C.L_EPOCHS:500,
        C.L_SHUFFLE:True,
        C.L_EPS: 0.1,
        C.L_DISTRIBUTION:C.UNIFORM,
        C.L_BIAS:0,
        C.L_SEED: 25,
        C.L_CLASSIFICATION:True,
        C.L_EARLYSTOP:False,
        C.L_PATIENCE:30,
        C.L_TRESHOLD_VARIANCE:1.e-8    
    }

    theta_mini = single_model.copy()
    theta_mini[C.L_ETA]= 0.09
    theta_mini[C.L_TAU]= (100, 0.009)
    theta_mini[C.L_DIMBATCH]= 25
    theta_mini[C.L_MOMENTUM]= (C.CLASSIC, 0.8)

    dirName="TestMonk_1"
    TR_x_monk1, TR_y_monk1 = readMC.get_train_Monk_1(single_model[C.L_ACTIVATION][-1]==C.TANH)
    TS_x_monk1, TS_y_monk1 = readMC.get_test_Monk_1(single_model[C.L_ACTIVATION][-1]==C.TANH)
    monk_model_evaluation(TR_x_monk1, TR_y_monk1, TS_x_monk1, TS_y_monk1, single_model, dirName, prefixFilename = C.PREFIXBATCH)
    monk_model_evaluation(TR_x_monk1, TR_y_monk1, TS_x_monk1, TS_y_monk1, theta_mini, dirName, prefixFilename = C.PREFIXMINIBATCH)
    
    dirName="TestMonk_2"
    theta_mini[C.L_ETA]= 0.02
    theta_mini[C.L_TAU]= (100, 0.002)
    theta_mini[C.L_MOMENTUM]= (C.NESTEROV, 0.8)
    TR_x_monk2, TR_y_monk2 = readMC.get_train_Monk_2(single_model[C.L_ACTIVATION][-1]==C.TANH)
    TS_x_monk2, TS_y_monk2 = readMC.get_test_Monk_2(single_model[C.L_ACTIVATION][-1]==C.TANH)
    monk_model_evaluation(TR_x_monk2, TR_y_monk2, TS_x_monk2, TS_y_monk2, single_model, dirName, prefixFilename = C.PREFIXBATCH)
    monk_model_evaluation(TR_x_monk2, TR_y_monk2, TS_x_monk2, TS_y_monk2, theta_mini, dirName, prefixFilename = C.PREFIXMINIBATCH)
    
    dirName="TestMonk_3"
    theta_mini[C.L_MOMENTUM]= (C.CLASSIC, 0.8)
    theta_mini[C.L_ETA]= 0.09
    theta_mini[C.L_TAU]= (100, 0.009)
    TR_x_monk3, TR_y_monk3 = readMC.get_train_Monk_3()
    TS_x_monk3, TS_y_monk3 = readMC.get_test_Monk_3()
    monk_model_evaluation(TR_x_monk3, TR_y_monk3, TS_x_monk3, TS_y_monk3, single_model, dirName, prefixFilename = C.PREFIXBATCH)
    monk_model_evaluation(TR_x_monk3, TR_y_monk3, TS_x_monk3, TS_y_monk3, theta_mini, dirName, prefixFilename = C.PREFIXMINIBATCH)
    
    single_model[C.L_REG]=(C.LASSO,0.001)
    monk_model_evaluation(TR_x_monk3, TR_y_monk3, TS_x_monk3, TS_y_monk3, single_model, dirName, prefixFilename = C.PREFIXBATCH+"REG")    
"""
if __name__ == "__main__":
    
   
    DIRNAME:str = "TestMonk_1"
    main()
    
    #TR_x_monk2, TR_y_monk2 = readMC.get_train_Monk_2()
    #TS_x_monk2, TS_y_monk2 = readMC.get_test_Monk_2()
    #DIRNAME:str = "TestMonk_2"
    #main(TR_x_monk2, TR_y_monk2, TS_x_monk2, TS_y_monk2, DIRNAME)

    """
    TR_x_monk3, TR_y_monk3 = readMC.get_train_Monk_3()
    TS_x_monk3, TS_y_monk3 = readMC.get_test_Monk_3()
    DIRNAME:str = "TestMonk_3"
    """  
   