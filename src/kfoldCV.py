from dnnPlot import draw_async
import numpy as np
import costants as C
from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from candidateHyperparameters import Candidate
from candidateHyperparameters import CandidatesHyperparameters
import time
import os

import kfoldLog
import gridSearch

class KfoldCV:
    def __init__(self, inputs, targets, k, candidates_hyperparameters = CandidatesHyperparameters()):
        self.history_error:list = []
        self.k:int = k   
        self.candidates_hyperparameters:CandidatesHyperparameters = candidates_hyperparameters
        self.inputs = inputs
        self.targets = targets
        self.kfolds:list = self.divide_dataset()
        if k <= 1:
            raise ValueError("The Number of k folds must be more than 1")    
        print(f"---- Dataset splitted in #{self.k} Fold ----")
        self.models_error:list = []

    # Compute the splitting in fold
    # :return: folds are all inputs for row of kfoldsCV
    def divide_dataset(self):
        #initialize empty list
        folds:list = []
        #split the arrays in  k folds        
        input_k = np.array_split(self.inputs, self.k)
        target_k = np.array_split(self.targets, self.k)
        #loop the pair of fold the indexed will be the validation, other the train
        for i, pair in enumerate(zip(input_k, target_k)):
            x_train = np.concatenate(input_k[:i] + input_k[i + 1:])
            y_train = np.concatenate(target_k[:i] + target_k[i + 1:])
            x_val = pair[0]
            y_val = pair[1]
            D_row = {
                C.INPUT_TRAINING : x_train,
                C.OUTPUT_TRAINING : y_train,
                C.INPUT_VALIDATION : x_val,
                C.OUTPUT_VALIDATION : y_val,
                C.K_FOLD : i + 1
            }
            #append the fold inside a list and return
            folds.append(D_row)
        return folds

    # For train the model in each fold and return the error
    # :param: fold is the dictionary, contain hyperparameters, and datasets of the fold 
    # :param: theta is the configuration object, contain hyperparameters 
    # :param: candidate number is the number of possible model
    # :param: drawPlot is a flag for draw the plot
    # :param: pathPlot is the path for the plot
    # :return: error a list with validation error of current fold
    @kfoldLog.timeit
    def train_fold(self, fold, theta, candidatenumber:int, drawPlot:bool = True, pathPlot:str = None):
        #recover hyperparameters
        x_train = fold[C.INPUT_TRAINING]
        y_train = fold[C.OUTPUT_TRAINING]
        x_val = fold[C.INPUT_VALIDATION]
        y_val = fold[C.OUTPUT_VALIDATION]
        print("THETA", theta)
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
            namefile = f"candidate{candidatenumber}fold{fold[C.K_FOLD]}"
            if pathPlot is not None:
                plot_path = f'../plot/{pathPlot}/{namefile}'
            else:
                plot_path = f'../plot/{time.strftime(C.FORMATTIMESTAMP)}/{namefile}'

            if model.classification:
                inYlim = (-0.5, 1.1)
                inMSELim = (0,0) 
            else:
                inMSELim = (0.2,(error[C.ERROR][-1]*5) )
                inYlim = (-0.5, 5.)

            # start = time.time()

            labelMetric = C.ACCURACY if theta[C.L_CLASSIFICATION] else 'MEE'
            process = draw_async(error[C.ERROR], error['validation'], error[C.METRIC_TR], error['metric_val'], error_tr = error[C.LOSS],
                        lbl_tr = C.LABEL_PLOT_TRAINING, lbl_vs = C.LABEL_PLOT_VALIDATION, path = plot_path, 
                        ylim = inYlim, yMSElim = inMSELim, titlePlot = f"Model \#{candidatenumber} fold {fold[C.K_FOLD]}",
                        theta = theta, labelsY = ['Loss', labelMetric])

            # end = time.time()
            # print(f'Plot Graph {end-start}')
        #endregion
        # print('train fold')
        # print(process)
        return error, process
    
    
    # Compute the error of a set of hyperparametric values and return the mean between the errors
    # :param: hyperparameters for estimation
    # :param: log is for define the logger
    # :param: inCandidatenumber is input candidate number
    # :param: plot is a flag for draw the plot
    # :param: pathPlot is the path for the plot
    # :return: processes for drawing plot
    @kfoldLog.timeit
    def estimate_model_error(self, hyperparameters, log = None, inCandidatenumber:int = 0, plot:bool = True, pathPlot:str = None):
        processes = []
        # t_mse Mean Square Error of the training data
        # v_mse Mean Square Error of the validation data
        # mae   Mean Absolute error
        # rmse  Root Mean Squared Error
        # mee   Mean Euclidean Error
        t_mse, v_mse, mae, rmse, mee, epochs ,t_accuracy, v_accuracy = 0, 0, 0, 0, 0, 0, 0, 0
        
        for fold in self.kfolds:
            errors, process = self.train_fold(fold, hyperparameters, candidatenumber = inCandidatenumber, drawPlot = plot, pathPlot = pathPlot)
            # print('estimate_model_error')
            # print(process)
            
            processes.append(process)
            h_train = errors[C.ERROR]
            h_validation = errors['validation']
            t_mse   += h_train[-1]
            v_mse   += h_validation[-1]
            mae     += errors['mean_absolute_error']
            rmse    += errors['root_mean_squared_error']
            mee     += errors['mean_euclidean_error']
            epochs  += errors[C.EPOCHS]
            if errors[C.C_METRICS][f'{C.VALIDATION}_{C.ACCURACY}']:
                t_accuracy += errors[C.C_METRICS][f'{C.TRAINING}_{C.ACCURACY}'][-1]
                v_accuracy += errors[C.C_METRICS][f'{C.VALIDATION}_{C.ACCURACY}'][-1]

        mean_train  = t_mse / self.k
        mean_validation = v_mse / self.k
        mean_mae    = mae / self.k
        mean_rmse = rmse / self.k
        mean_mee = mee / self.k
        mean_epochs = epochs / self.k

        model_error = {
            C.CANDIDATE_NUMBER: inCandidatenumber,
            C.HYPERPARAMETERS: hyperparameters,
            "mean_train": mean_train,
            "mean_mae": mean_mae,
            'mean_validation':mean_validation  ,
            "mean_rmse":mean_rmse,
            'mean_mee':mean_mee,
            'mean_epochs':mean_epochs,
        }

        if errors[C.C_METRICS][f'{C.VALIDATION}_{C.ACCURACY}']:
            mean_t_accuracy = t_accuracy / self.k
            mean_v_accuracy = v_accuracy / self.k
            model_error[f'mean_{C.TRAINING}_{C.ACCURACY}'] = mean_t_accuracy
            model_error[f'mean_{C.VALIDATION}_{C.ACCURACY}'] = mean_v_accuracy

        kfoldLog.model_performance(log, hyperparameters, model_error)
        self.models_error.append(model_error)
        # print('estimate_model_error')
        # print(processes)
        return processes
        
    # Compute the best model
    # :return: the model with the best estimated error
    @kfoldLog.timeit
    def the_winners_are(self, num_best_models:int = C.NUM_WINNER):
        means = [
            result["mean_mee"]
            for result in self.models_error
            if (not np.isnan(result["mean_mee"]))
        ]

        # choose the set of hyperparameters which gives the minimum mean error
        means_length = len(means)
        if(means_length == 0):
            raise ValueError('Nobody is the winner')
        means.sort()
        self.models_error = sorted(self.models_error, key = lambda d: d['mean_mee'] if not np.isnan(d['mean_mee']) else 1.e+10, reverse = False)
        winners_list = []

        if(num_best_models > means_length):
            num_best_models = means_length
        for index in range(num_best_models):
            meanmetrics = {key: self.models_error[index][key] for key in ['mean_train','mean_validation','mean_mae','mean_rmse', 'mean_mee','mean_epochs']}
            winner = {
                C.HYPERPARAMETERS:self.models_error[index][C.HYPERPARAMETERS], 
                C.CANDIDATE_NUMBER:self.models_error[index][C.CANDIDATE_NUMBER],
                C.MEAN_METRICS: meanmetrics
            }
            winners_list.append(winner)
        return winners_list
    
    # Execute tht Kfold validation
    # :param: inDefault is the input default configuration name 
    # :param: inTheta is the input configuration object
    # :param: FineGS is the flag for enable the fine-grained validation
    # :param: plot is a flag for draw the plot
    # :param: prefixFilename is the prefix to add to the file name for manage the dirctory and the produced plots
    # :param: clearOldThetaWinner is for clear old theta of old winner
    # :return: the winner of the validation and its mean metrics
    @kfoldLog.timeit
    def validate(self, inDefault:str = C.MONK, inTheta = None, FineGS:bool = False, plot:bool = True, prefixFilename:str = "", clearOldThetaWinner:bool = True):
        processes = []
        if clearOldThetaWinner:
            self.models_error.clear()
        self.candidates_hyperparameters.set_project_hyperparameters(default = inDefault, theta = inTheta)
        
        """ K-Fold Cross Validation """
        # a first coarse Grid Search, values differ in order of magnitude
        all_candidates, total = gridSearch.grid_search(hyperparameters = self.candidates_hyperparameters)
        if len(all_candidates.get_all_candidates_dict()) == 0:
            print("The new candidates were not created, probably the activate functions were setted very bad")
        log, timestr = kfoldLog.start_log(f"{prefixFilename}_ModelSelection")
        
        # Directory
        new_directory_name:str = f"{prefixFilename}{C.PREFIX_DIR_COARSE}{timestr}"
        # Parent Directory path
        
        # Path
        path_dir_models_coarse = os.path.join(C.PATH_PLOT_DIR, new_directory_name)
        if not os.path.exists(path_dir_models_coarse):
            os.makedirs(path_dir_models_coarse)
        for i, theta in enumerate(all_candidates.get_all_candidates_dict()):
            kfoldLog.estimate_model(log, i+1, total)
            inProcesses = self.estimate_model_error(theta, log, inCandidatenumber = i+1, plot = plot, pathPlot = path_dir_models_coarse)
            # print('validate')
            # print(inProcesses)
            # input('premi')
            if(not C.UNIX):
                processes.extend(inProcesses)
            if(len(processes) > 20):
                clearProcesses(processes)
        clearProcesses(processes)
        
        
        winners_list = self.the_winners_are()
        winner = Candidate(winners_list[0][C.HYPERPARAMETERS])
        kfoldLog.the_winner_is(log, winners_list[0][C.CANDIDATE_NUMBER], winner.to_string())
        kfoldLog.end_log(log)

        if FineGS:
            processes = []
            self.models_error.clear()
            log, timestr = kfoldLog.start_log(f"{prefixFilename}_FineModelSelection")
            possible_winners, total = gridSearch.grid_search(hyperparameters = winner, coarse = False)
            print("---Start Fine Grid search...\n")
            # Directory
            directory = f"{prefixFilename}{C.PREFIX_DIR_FINE}{timestr}"
            # Parent Directory path
            parent_dir = "../plot/"
            # Path
            path_dir_models_fine = os.path.join(parent_dir, directory)
            if not os.path.exists(path_dir_models_fine):
                os.makedirs(path_dir_models_fine)
        
            for j, theta in enumerate(possible_winners.get_all_candidates_dict()):
                kfoldLog.estimate_model(log, j+1, total)
                inProcesses = self.estimate_model_error(theta, log, inCandidatenumber = j+1, plot = plot, pathPlot = path_dir_models_fine)
                if(not C.UNIX):
                    processes.extend(inProcesses)
                if(len(processes) > 20):
                    clearProcesses(processes)
            winners_list = self.the_winners_are()
            for winner in winners_list:
                kfoldLog.the_fine_winner_is(log, winner[C.CANDIDATE_NUMBER], Candidate(winner[C.HYPERPARAMETERS]).to_string(), metric = f"MeanMee: {winner[C.MEAN_METRICS]['mean_mee']}")  
            kfoldLog.end_log(log)
        return winners_list
    
def clearProcesses(processes):
    if(not C.UNIX):
        for process in processes:
            if(process != None):
                # print('join')
                process.join()
            else:
                print('The process is None, Something is wrong')
        processes = []