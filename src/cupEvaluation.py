# Evaluate the cup dataset
from candidateHyperparameters import Candidate
from dnnPlot import draw_async
from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV
import costants as C
from dnnPlot import draw_async
import os
import kfoldLog
import readMonkAndCup as readMC
import numpy as np

# :param: TR_x_cup is training dataset
# :param: TR_y_cup is training targets dataset
# :param: TS_x_cup is test dataset
# :param: TS_y_cup is test targets dataset
# :param: theta is configuration object
# :param: dirName is directory name
# :param: prefixFilename is prefix of the file name
# :param: fold is number of the folds
def cup_evaluation(TR_x_cup, TR_y_cup, TS_x_cup, TS_y_cup, theta, dirName, prefixFilename, fold = 2,FineGS=True):
    #MODEL SELECTION KFCV
    kfCV = KfoldCV(TR_x_cup, TR_y_cup, fold)
    winners_list = kfCV.validate(inTheta =  theta, FineGS = FineGS, prefixFilename = prefixFilename)
    
    for winner_object in winners_list:
        winner = Candidate(winner_object[C.HYPERPARAMETERS])
        winnerTheta = winner.get_dictionary()

        #MODEL ASSESSMENT HOLDOUT
        trerrors, euclidianAccuracy, results = holdoutTest(winnerTheta, TR_x_cup, TR_y_cup, TS_x_cup, TS_y_cup, val_per = 0.25, meanepochs = int(winner_object[C.MEAN_METRICS]['mean_epochs']))
        _, timestamp = kfoldLog.Model_Assessment_log(dirName, prefixFilename, f"Model Hyperparameters:\n {str(winnerTheta)}\n", f"Model Selection Result obtained in {fold}# folds:\n{winner_object[C.MEAN_METRICS]}\n Mean Euclidian Error:\n{euclidianAccuracy}\n")
        kfoldLog.Model_Assessment_Outputs(results, dirName, f'{prefixFilename}_HouseTest', timestamp = timestamp)
        savePlotFig(trerrors, dirName, prefixFilename, f"{dirName}{prefixFilename}", theta = winnerTheta)

        #BLIND TEST
        TS_x_CUP_blind = readMC.get_blind_test_CUP()
        TR_x_CUP_All, TR_y_CUP_All,_,_ = readMC.get_cup_house_test(perc = 0)
        _, _, resultsBlind = holdoutTest(winnerTheta,  TR_x_CUP_All, TR_y_CUP_All, TS_x_CUP_blind, [], val_per =  0.25, meanepochs = int(winner_object[C.MEAN_METRICS]['mean_epochs']))
        kfoldLog.ML_Cup_Template(resultsBlind, dirName, f'{prefixFilename}_blind', timestamp)
    return winners_list

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

# Takes some the best winners models and computes the mean errors and outputs
# :param: models is the list of the configurations
# :param: tr_x is training dataset
# :param: tr_y is training targets dataset
# :param: TS_x_cup is test dataset if i want to use a specific dataset instead it reads the blind dataset
# :param: TS_y_cup is test targets dataset if it has a output dataset for check the error between targets and mean outputs
# :param: dirname is the name of the directory where will be the file with ensemble
# :param: filename is the name of the file was produced by method
def ensemble_Cup(models, tr_x, tr_y, Ts_x=None, Ts_y=None, dirname = "Ensamble_cup", filename = "CUPs"):
    if Ts_x is None:
        Ts_x = []
    if Ts_y is None:
        Ts_y = []
    inputs = readMC.get_blind_test_CUP() if (isinstance(Ts_x, list)) else Ts_x
    outs = []
    errors = []
    for model in models:
        model = dnn(**model[C.HYPERPARAMETERS])
        errors.append(model.fit(tr_x, tr_y))
        outs.append(model.forward_propagation(inputs))
    #BEFORE MEANS of OUTPUT and THEN EVALUATION OF THE METRICS
    mean_outs = np.mean(outs, axis = 0)
    #CHECK IF BLIND 
    if (not isinstance(Ts_y, list)):
        euclidianAccuracy = model.metrics.mean_euclidean_error(Ts_y, mean_outs)
        result = np.concatenate((Ts_y, mean_outs), axis=1)
        print("Ensembe EuclidianError:", euclidianAccuracy)
        kfoldLog.Model_Assessment_Outputs(result, dirname, f'{filename}_MEE_{euclidianAccuracy}')
    else:
        result = mean_outs
        euclidianAccuracy = None
        kfoldLog.ML_Cup_Template(result, dirname, f'{filename}_Blind')
   
# Makes the plots of the test in multi processing
# :param: errors is the list of the computed errors
# :param: dirname is the name of the directory where will be the new plot file
# :param: filename is the name of the file was produced by method
# :param: title is the title on the plot
# :param: theta is the configuration object for add in the subtitle of the plot
def savePlotFig(errors, dirName, fileName, title, theta):
    # Path
    path_dir_models_coarse = os.path.join(C.PATH_PLOT_DIR, dirName)
    if not os.path.exists(path_dir_models_coarse):
            os.makedirs(path_dir_models_coarse)
    # is false if the loss is zero else take the loss 
    inError_tr = False if errors[C.LOSS] == 0 else errors[C.LOSS]
    labelError = 'validation'
    metric = 'metric_val'
    if len(errors['test']) > 0:
        labelError = 'test'
        metric = 'metric_test'
    process = draw_async(errors[C.ERROR], errors[labelError], errors[C.METRIC_TR], errors[metric], error_tr = inError_tr,
                        lbl_tr = C.LABEL_PLOT_TRAINING, lbl_vs = labelError.capitalize(), path = f"{path_dir_models_coarse}/{fileName}", 
                        ylim = (-0.5, 10),yMSElim=(0,(errors[C.ERROR][-1])*100) ,titlePlot = title,
                        theta = theta, labelsY = ['Loss',  "MEE"])
    if(process != None and not C.UNIX):
        process.join()
     