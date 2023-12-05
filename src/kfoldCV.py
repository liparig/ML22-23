from dnn_plot import plot_curves
import numpy as np
from costants import FORMATTIMESTAMP, LABEL_PLOT_TRAINING, LABEL_PLOT_VALIDATION, PATH_PLOT_DIR, PREFIX_DIR_COARSE
from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from candidate_hyperparameters import Candidate
from candidate_hyperparameters import Candidates_Hyperparameters
import time
import os

import kfoldLog
import grid_search

class KfoldCV:
    def __init__(self, inputs, targets, k, candidates_hyperparameters = Candidates_Hyperparameters()):
        self.history_error:list = []
        self.k:int = k   
        self.candidates_hyperparameters:Candidates_Hyperparameters = candidates_hyperparameters
        self.inputs = inputs
        self.targets = targets
        self.kfolds:list = self.divide_dataset()
        print(f"---- Dataset diviso in #{self.k} Fold ----")
        self.models_error:list = []

    '''
    Compute the splitting in fold and build a dictionary with the hyperparameters
    :param hyperparameters: theta for model selection
    :return folds: all inputs for row of kfoldsCV
    '''
    def divide_dataset(self):
        #initialize empty list
        folds:list = []
        #split the arrays in  k folds        
        input_k = np.array_split(self.inputs, self.k)
        target_k = np.array_split(self.targets, self.k)
        #loop the pair of fold the indexed will be the validation, other the train
        for i, pair in enumerate(zip(input_k,target_k)):
            x_train = np.concatenate(input_k[:i] + input_k[i + 1:])
            y_train = np.concatenate(target_k[:i] + target_k[i + 1:])
            x_val = pair[0]
            y_val = pair[1]
            D_row = {
                "x_train" : x_train,
                "y_train" : y_train,
                "x_val" : x_val,
                "y_val" : y_val,
                "k" : i + 1
            }
            #append the fold inside a list and return
            folds.append(D_row)
        return folds

    '''
    For train the model in each fold and return the error
    :param fold: dictionary, contain hyperparameters, and datasets of the fold 
    :return error: a list with validation error of current fold
    '''
    def train_fold(self, fold, theta, candidatenumber:int, drawPlot:bool = True, pathPlot:str = None):
        #recover hyperparameters
        x_train = fold["x_train"]
        y_train = fold["y_train"]
        x_val = fold["x_val"]
        y_val = fold["y_val"]
        
        model = dnn(**theta)
        #train
        error = model.fit(x_train, y_train, x_val, y_val)
        out = model.forward_propagation(x_val)
        error['mean_absolute_error'] = model.metrics.mean_absolute_error(y_val, out)
        error['root_mean_squared_error'] = model.metrics.root_mean_squared_error(y_val, out)
        error['mean_euclidean_error'] = model.metrics.mean_euclidean_error(y_val, out)

        #region Plot
        if drawPlot:
            #new model train from scratch
            namefile = f"candidate{candidatenumber}fold{fold['k']}"
            if pathPlot != None:
                plot_path = f'../plot/{pathPlot}/{namefile}'
            else:
                plot_path = f'../plot/{time.strftime(FORMATTIMESTAMP)}/{namefile}'
                
            if model.classification:
                inYlim = (-0.5, 1.5)
            else:
                inYlim = (-0.5, 5.)

            
            # return {'error':history_terror,'loss':history_tloss, 'mee':metric_tr, 'mee_v':metric_val, 'validation':validation_error, 'c_metrics':c_metric, 'epochs':epoch + 1}  
            # print(f"{theta}")
            # input()
            listModelTheta:list = f'{theta}'.split(', ') 
            numElems = len(listModelTheta)
            listModelTheta[len(listModelTheta)/4] = f'{listModelTheta[len(listModelTheta)/4]}\n'   
            listModelTheta[len(listModelTheta)/2] = f'{listModelTheta[len(listModelTheta)/2]}\n'   
            listModelTheta[len(listModelTheta)] = f'{listModelTheta[len(listModelTheta)]}\n'
            plot_curves(error['error'], error['validation'], error['mee'], error['mee_v'], 
                        lbl_tr = LABEL_PLOT_TRAINING, lbl_vs = LABEL_PLOT_VALIDATION, path = plot_path, 
                        ylim = inYlim, titleplot = f"Model #{candidatenumber} fold {fold['k']}",
                        titlesSubplot = [listModelTheta[:numElems/2],listModelTheta[(numElems/2)+1:]])
        #endregion
        
        return error
    
    '''
    Compute the error of a set of hyperparametric values and return the mean between the errors
    :param hyperparameters: hyperparameters for estimation
    :return error_mean: means of the different metrics validation error
    '''
    def estimate_model_error(self, hyperparameters, log = None, inCandidatenumber:int = 0, plot:bool = True, path:str = None):
        '''
        t_mse Mean Square Error of the training data
        v_mse Mean Square Error of the validation data
        mae   Mean Absolute error
        rmse  Root Mean Squared Error
        mee   Mean Euclidean Error
        '''
        t_mse, v_mse, mae, rmse, mee, epochs ,t_accuracy,v_accuracy= 0, 0, 0, 0, 0, 0, 0, 0
        #d_row = self.divide_dataset(hyperparameters)
        varianceMSE = []
        for fold in self.kfolds:
            #print(f"- Fold {i} ",end="")
            errors = self.train_fold(fold, hyperparameters, candidatenumber = inCandidatenumber, drawPlot = plot, pathPlot = path)
            h_train = errors['error']
            h_validation = errors['validation']
            varianceMSE.append(h_validation)
            t_mse   += h_train[-1]
            v_mse   += h_validation[-1]
            mae     += errors['mean_absolute_error']
            rmse    += errors['root_mean_squared_error']
            mee     += errors['mean_euclidean_error']
            epochs  += errors['epochs']
            if errors['c_metrics']['v_accuracy']:
                t_accuracy += errors['c_metrics']['t_accuracy'][-1]
                v_accuracy += errors['c_metrics']['v_accuracy'][-1]

        mean_train  = t_mse / self.k
        mean_validation = v_mse / self.k
        mean_mae    = mae / self.k
        mean_rmse = rmse / self.k
        mean_mee = mee / self.k
        mean_epochs = epochs / self.k
        #print(f"\nMean MEE: {mean_mee} - Mean MSE {mean_validation} - MeanEpochs: {mean_epochs}")
        model_error = {
            "candidateNumber": inCandidatenumber,
            "hyperparameters": hyperparameters,
            "mean_train": mean_train,
            "mean_mae": mean_mae,
            'mean_validation':mean_validation  ,
            "mean_rmse":mean_rmse,
            'mean_mee':mean_mee,
            'mean_epochs':mean_epochs,
            'varianceMSE':varianceMSE
        }

        if errors['c_metrics']['v_accuracy']:
            mean_t_accuracy = t_accuracy / self.k
            mean_v_accuracy = v_accuracy / self.k
            model_error['mean_t_accuracy'] = mean_t_accuracy
            model_error['mean_v_accuracy'] = mean_v_accuracy
            #print( f"Classification Accuracy Training: {mean_t_accuracy} - Validation {mean_v_accuracy}")
        kfoldLog.model_performance(log, hyperparameters, model_error)
        self.models_error.append(model_error)
        return v_mse
    '''
    :return: the model with the best estimated error
    '''
    def the_winner_is(self, classification = True):
        means = []
        for result in self.models_error:
            if classification:
                means.append(-result["mean_v_accuracy"])
            else:
                means.append(result["mean_mee"])
        # choose the set of hyperparameters which gives the minimum mean error
        lower_mean = np.argmin(means)
        return self.models_error[lower_mean]["hyperparameters"], self.models_error[lower_mean]["candidateNumber"]
    
    def validate(self, default:str = "monk", FineGS:bool = False):
        if default == "monk" or default == "cup":
            self.candidates_hyperparameters.set_project_hyperparameters(default)
        """ K-Fold Cross Validation """
        # a first coarse Grid Search, values differ in order of magnitude
        create_candidate, total = grid_search.grid_search(hyperparameters = self.candidates_hyperparameters)
        log, timestr = kfoldLog.start_log("ModelSelection")
        
        # Directory
        new_directory_name:str = f"{PREFIX_DIR_COARSE}{timestr}"
        # Parent Directory path
        
        # Path
        path_dir_models = os.path.join(PATH_PLOT_DIR, new_directory_name)
        if not os.path.exists(path_dir_models):
            os.makedirs(path_dir_models)
        
        for i,theta in enumerate(create_candidate.get_all_candidates_dict()):
            kfoldLog.estimate_model(log, i+1, total)
            
            self.estimate_model_error(theta, log, inCandidatenumber = i+1, path = path_dir_models)
            
        winner, modelnumber = self.the_winner_is(classification = self.candidates_hyperparameters.classification[0])
        winner = Candidate(winner)
        kfoldLog.the_winner_is(log,modelnumber,winner.to_string())
        kfoldLog.end_log(log)

        if FineGS:
            log,timestr = kfoldLog.start_log("FineModelSelection")
            possible_winners, total = grid_search.grid_search(hyperparameters = winner, coarse = False)
            print("---Start Fine Grid search...\n")
            # Directory
            directory = f"Fine{timestr}"
            # Parent Directory path
            parent_dir = "../plot/"
            # Path
            path = os.path.join(parent_dir, directory)
            if not os.path.exists(path):
                os.makedirs(path)
        
            for j,theta in enumerate(possible_winners.get_all_candidates_dict()):
                kfoldLog.estimate_model(log, j+1, total)
                meanMSE = self.estimate_model_error(theta, log, inCandidatenumber = j+1, path = path_dir_models)

            true_winner,modelnumber = self.the_winner_is(classification = self.candidates_hyperparameters.classification[0])
            true_winner = Candidate(true_winner)
            kfoldLog.the_fine_winner_is(log,modelnumber,winner.to_string(),metric=f"MeanMee: {meanMSE}")
            kfoldLog.end_log(log)
        return true_winner 