import numpy as np
from costants import FORMATTIMESTAMP
from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from candidate_hyperparameters import Candidate
from candidate_hyperparameters import Candidates_Hyperparameters
import time
import os


import grid_search

class KfoldCV:
    def __init__(self, inputs, targets, kfolds, candidates_hyperparameters = Candidates_Hyperparameters()):
        self.history_error:list = []
        self.kfolds:int = kfolds
        self.candidates_hyperparameters:Candidates_Hyperparameters = candidates_hyperparameters
        self.inputs = inputs
        self.targets = targets
        self.models_error:list = []

    '''
    Compute the splitting in fold and build a dictionary with the hyperparameters
    :param hyperparameters: theta for model selection
    :return folds: all inputs for row of kfoldsCV
    '''
    def divide_dataset(self, hyperparameters):
        #initialize empty list
        folds:list = []
        #split the arrays in  k folds        
        input_k = np.array_split(self.inputs, self.kfolds)
        target_k = np.array_split(self.targets, self.kfolds)
        dim_batch = hyperparameters.pop("dim_batch")
        #loop the pair of fold the indexed will be the validation, other the train
        for i, pair in enumerate(zip(input_k,target_k)):
            x_train = np.concatenate(input_k[:i] + input_k[i + 1:])
            y_train = np.concatenate(target_k[:i] + target_k[i + 1:])
            x_val = pair[0]
            y_val = pair[1]
            D_row = {
                'hyperparameters':hyperparameters,
                "x_train" : x_train,
                "y_train" : y_train,
                "x_val" : x_val,
                "y_val" : y_val,
                "dim_batch" : dim_batch,
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
    def train_fold(self, fold, candidatenumber,timestr):
        #recover hyperparameters
        x_train = fold["x_train"]
        y_train = fold["y_train"]
        x_val = fold["x_val"]
        y_val = fold["y_val"]
        hyperparameters = fold["hyperparameters"]
        dim_batch = fold["dim_batch"]
        #new model train from scratch
        namefile = f"candidate{candidatenumber}fold{fold['k']}"
        
        # Directory
        directory = str(timestr)
        # Parent Directory path
        parent_dir = "../plot/"
        # Path
        path = os.path.join(parent_dir, directory)
        if not os.path.exists(path):
            os.makedirs(path)
        plot_path = f'../plot/{timestr}/{namefile}'
        model = dnn(**hyperparameters, plot = plot_path)
        #train
        error = model.fit(x_train, y_train, x_val, y_val, dim_batch)
        out = model.forward_propagation(x_val)
        error['mean_absolute_error'] = model.metrics.mean_absolute_error(y_val, out)
        error['root_mean_squared_error'] = model.metrics.root_mean_squared_error(y_val, out)
        error['mean_euclidean_error'] = model.metrics.mean_euclidean_error(y_val, out)

        return error
    
    '''
    Compute the error of a set of hyperparametric values and return the mean between the errors
    :param hyperparameters: hyperparameters for estimation
    :return error_mean: means of the different metrics validation error
    '''
    def estimate_model_error(self, hyperparameters, file = None, inCandidatenumber = 0, **kwargs):
        '''
        t_mse Mean Square Error of the training data
        v_mse Mean Square Error of the validation data
        mae   Mean Absolute error
        rmse  Root Mean Squared Error
        mee   Mean Euclidean Error
        '''
        t_mse, v_mse, mae, rmse, mee, epochs = 0, 0, 0, 0, 0, 0
        d_row = self.divide_dataset(hyperparameters)
        dim_batch = 0
        varianceMSE = []
        for d in d_row:
            errors = self.train_fold(d, candidatenumber = inCandidatenumber, **kwargs)
            h_train = errors['error']
            h_validation = errors['validation']
            varianceMSE.append(h_validation)
            t_mse   += h_train[-1]
            v_mse   += h_validation[-1]
            mae     += errors['mean_absolute_error']
            rmse    += errors['root_mean_squared_error']
            mee     += errors['mean_euclidean_error']
            epochs  += errors['epochs']
            dim_batch = d["dim_batch"]
        number_of_d_rows:int = len(d_row)
        mean_train  = t_mse / number_of_d_rows
        mean_validation = v_mse / number_of_d_rows
        mean_mae    = mae / number_of_d_rows
        mean_rmse = rmse / number_of_d_rows
        mean_mee = mee / number_of_d_rows
        mean_epochs = epochs / number_of_d_rows
        hyperparameters['dim_batch'] = dim_batch
        model_error = {
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
            t_accuracy = errors['c_metrics']['t_accuracy'][-1]
            v_accuracy = errors['c_metrics']['v_accuracy'][-1]
        
        s = f" Model: {hyperparameters} \n Mean Train: {model_error['mean_train']} \n Mean Mae {model_error['mean_mae']}"\
            f"\n Mean Validation: {model_error['mean_validation']} \n Train Accuracy: {t_accuracy} \nValidation Accuracy: {v_accuracy}"\
            f"\n Mean Rmse: {model_error['mean_rmse']}"\
            f"\n Mean MEE: {model_error['mean_mee']}"\
            f"\n mean Epochs: {model_error['mean_epochs']}"
        if file:
            file.write(s)
        else:
            print(s)

        self.models_error.append(model_error)
        return v_mse
    '''
    :return: the model with the best estimated error
    '''
    def the_winner_is(self):
        means = []
        theta = []
        for result in self.models_error:
            means.append(result["mean_validation"])
            theta.append(result["hyperparameters"])
        # choose the set of hyperparameters which gives the minimum mean error
        lower_mean = np.argmin(means)
        self.models_error
        return theta[lower_mean]

    
    def validate(self, default:str = "monk", FineGS:bool = False):
        if default == "monk" or default == "cup":
            self.candidates_hyperparameters.set_project_hyperparameters(default)
        """ K-Fold Cross Validation """
        # a first coarse Grid Search, values differ in order of magnitude
        create_candidate, total = grid_search.grid_search(hyperparameters = self.candidates_hyperparameters)
            # Writing to file
        timestr:str = time.strftime(FORMATTIMESTAMP)
        if(not(os.path.isdir('../KFoldCV'))):
            os.makedirs('../KFoldCV')
        with open(f"../KFoldCV/Gridsearch{timestr}.txt", "w") as file1:
            for i,theta in enumerate(create_candidate.get_all_candidates_dict()):
                print(f"----\n Estimate error for model #{i} of {total} \n----")
                file1.write(f"----\n Estimate error for model #{i} of {total} \n----")
                self.estimate_model_error(theta, file = file1, inCandidatenumber = i, timestr = f"Coarse{timestr}")
            
            winner:Candidate = Candidate(self.the_winner_is())
            print(f"---THE WINNER IS...\n {winner.to_string()}")
            file1.write(f"---THE WINNER IS...\n {winner.to_string()}")
            file1.write("Try to do better... with fine grid search")

        if FineGS:
            timestr = time.strftime(FORMATTIMESTAMP)
            with open(f"../KFoldCV/FineGridSearch{timestr}.txt", "w") as file2:
                possible_winners, total = grid_search.grid_search(hyperparameters = winner, coarse = False)
                print("---Start Fine Grid search...\n")
                for i,theta in enumerate(possible_winners.get_all_candidates_dict()):
                    print(f"----\nEstimate error for model #{i} of {total}\n----")
                    file2.write(f"----\nEstimate error for model #{i} of {total}\n----")
                    meanMSE = self.estimate_model_error(theta, file = file2, inCandidatenumber = i, timestr = f"Fine{timestr}")
                true_winner = Candidate(self.the_winner_is())
                print(f"---THE WINNER IS...\n {true_winner.to_string()}")
                file2.write(f"---THE GS WINNER IS...\n {winner.to_string()} \n with meanMSE: {meanMSE}")    
        return true_winner 