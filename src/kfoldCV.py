import numpy as np
from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from candidate_hyperparameters import Candidate
from candidate_hyperparameters import Candidates_Hyperparameters
import time
import os


import grid_search

class KfoldCV:
    candidates=Candidates_Hyperparameters()
    def __init__(self,inputs,targets, kfolds, candidates_hyperparameters=candidates):
        self.history_error=[]
        self.kfolds=kfolds
        self.candidates_hyperparameters=candidates_hyperparameters
        self.inputs=inputs
        self.targets=targets
        self.models_error=[]

    '''
    Compute the splitting in fold and build a dictionary with the hyperparameters
    :param hyperparameters: theta for model selection
    :return folds: all inputs for row of kfoldsCV
    '''
    def divide_dataset(self,hyperparameters):
        #initialize empty list
        folds=[]
        #split the arrays in  k folds        
        input_k=np.array_split(self.inputs, self.kfolds)
        target_k=np.array_split(self.targets, self.kfolds)
        dim_batch=hyperparameters.pop("dim_batch")
        #loop the pair of fold the indexed will be the validation, other the train
        for i,pair in enumerate(zip(input_k,target_k)):
            x_train= np.concatenate(input_k[:i] + input_k[i + 1:])
            y_train = np.concatenate(target_k[:i] + target_k[i + 1:])
            x_val=pair[0]
            y_val=pair[1]
            D_row={
                'hyperparameters':hyperparameters,
                "x_train": x_train,
                "y_train": y_train,
                "x_val":x_val,
                "y_val":y_val,
                "dim_batch":dim_batch,
                "k":i+1
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
        y_train =fold["y_train"]
        x_val =fold["x_val"]
        y_val =fold["y_val"]
        hyperparameters = fold["hyperparameters"]
        k=fold["k"]
        dim_batch=fold["dim_batch"]
        #new model train from scratch
        namefile="candidato"+str(candidatenumber)+"fold"+str(k)
        
        # Directory
        directory = str(timestr)
        # Parent Directory path
        parent_dir = "../plot/"
        # Path
        path = os.path.join(parent_dir, directory)
        if not os.path.exists(path):
            os.mkdir(path)
        plot_path=f'../plot/{timestr}/{namefile}'
        model = dnn(**hyperparameters,plot=plot_path)
        #train
        error=model.fit(x_train, y_train, x_val, y_val,dim_batch)
        out=model.forward_propagation(x_val)
        error['mean_absolute_error']=model.metrics.mean_absolute_error(y_val,out)
        error['root_mean_squared_error']=model.metrics.root_mean_squared_error(y_val,out)
        error['mean_euclidean_error']=model.metrics.mean_euclidean_error(y_val,out)
       

        return error
    
    '''
    Compute the error of a set of hyperparametric values and return the mean between the errors
    :param hyperparameters: hyperparameters for estimation
    :return error_mean: means of the different metrics validation error
    '''
    def estimate_model_error(self, hyperparameters,file=None,**kwargs):
        t_mse,v_mse,mae,rmse,mee,epochs=0,0,0,0,0,0
        d_row=self.divide_dataset(hyperparameters)
        dim_batch=0
        varianceMSE=[]
        for d in d_row:
            errors=self.train_fold(d,**kwargs)
            h_train=errors['error']
            h_validation=errors['validation']
            varianceMSE.append(h_validation)
            t_mse+=h_train[-1]
            v_mse+=h_validation[-1]
            mae     += errors['mean_absolute_error']
            rmse    += errors['root_mean_squared_error']
            mee     += errors['mean_euclidean_error']
            epochs  += errors['epochs']
            dim_batch=d["dim_batch"]
        mean_train  = t_mse / len(d_row)
        mean_validation = v_mse  / len(d_row)
        mean_mae    = mae / len(d_row)
        mean_rmse = rmse / len(d_row)
        mean_mee = mee / len(d_row)
        mean_epochs = epochs/ len(d_row)
        mean_Allerrors    =(mean_validation+mean_mae+mean_rmse+mean_mee)/4
        hyperparameters['dim_batch']=dim_batch
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
            t_accuracy=errors['c_metrics']['t_accuracy'][-1]
            v_accuracy=errors['c_metrics']['v_accuracy'][-1]
        else: 
            accuracy=None
        s="Model: "+str(hyperparameters)+"\n Mean Train:"+str(model_error['mean_train'])+"\n Mean Mae"+str(model_error['mean_mae'])+\
            "\n Mean Validation:" + str(model_error['mean_validation'])+"\n Train Accuracy: "+str(t_accuracy)+"\nValidation Accuracy: "+str(v_accuracy)+\
            "\n Mean Rmse:" + str(model_error['mean_rmse'])+\
            "\n Mean MEE:" +str( model_error['mean_mee'])+ \
            "\n Mean Errors:" +str(mean_Allerrors)+\
            "\n mean Epochs:"+ str(model_error['mean_epochs'])
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
        for result in  self.models_error:
            means.append(result["mean_errors"])
            theta.append(result["hyperparameters"])
        # choose the set of hyperparameters which gives the minimum mean error
        lower_mean = np.argmin(means)
        self.models_error
        return theta[lower_mean]

    
    def validate(self, default="",FineGS=False):
        if default == "monk" or "cup":
            self.candidates_hyperparameters.set_project_hyperparameters(default)
        """ K-Fold Cross Validation """
        # a first coarse Grid Search, values differ in order of magnitude
        create_candidate,total= grid_search.grid_search(hyperparameters=self.candidates_hyperparameters, coarse=True)
            # Writing to file
        timestr = time.strftime("%d%m%Y-%H%M")
        with open("../KFoldCV/Gridsearch"+timestr+".txt", "w") as file1:
            for i,theta in enumerate(create_candidate.get_all_candidates_dict()):
                print("----\nEstimate error for model #"+str(i)+" of "+str(total)+"\n----")
                file1.write("----\nEstimate error for model #"+str(i)+" of "+str(total)+"\n----")
                self.estimate_model_error(theta,file1,candidatenumber=i,timestr="Coarse"+timestr)
            
            winner=Candidate(self.the_winner_is())
            print("---THE WINNER IS...\n",winner.to_string())
            file1.write("---THE WINNER IS...\n"+winner.to_string())
            file1.write("Try to do better... with fine grid search")

        if FineGS:
            timestr = time.strftime("%d%m%Y-%H%M")
            with open("../KFoldCV/FineGridSearch"+timestr+".txt", "w") as file2:
                true_winner,total = grid_search.grid_search(hyperparameters=winner,coarse=False)
                print("---Start Fine Grid search...\n")
                for i,theta in enumerate(true_winner.get_all_candidates_dict()):
                    print("----\nEstimate error for model #"+str(i)+" of "+str(total)+"\n----")
                    file2.write("----\nEstimate error for model #"+str(i)+" of "+str(total)+"\n----")
                    meanMSE=self.estimate_model_error(theta,file2,candidatenumber=i,timestr="Fine"+timestr)
                truewinner=Candidate(self.the_winner_is())
                print("---THE WINNER IS...\n",truewinner.to_string())
                file2.write("---THE GS WINNER IS...\n"+winner.to_string()+"\n with meanMSE:"+ str(meanMSE))    
        return truewinner 