from io import TextIOWrapper
import costants as C
import time
import os

      
def start_log(fileName:str):
        # Writing to file
        timestr:str = time.strftime(C.FORMATTIMESTAMP)
        if(not(os.path.isdir(C.PATH_KFOLDCV_DIR))):
            os.makedirs(C.PATH_KFOLDCV_DIR)
        msfile = open(f"{C.PATH_KFOLDCV_DIR}/{fileName}_{timestr}.txt", "w")
        return msfile, timestr

def estimate_model(file:TextIOWrapper, index:int, total:int, stdoutput:bool = True, txt:bool = True):
    msg = f"---->\nEstimate error for model #{index} of {total}"
    if(stdoutput):
        print(msg)
    if(txt):
        file.write(f"{msg}\n")  

def the_winner_is(file:TextIOWrapper, index:int, winner:str, stdoutput:bool = True, txt:bool = True):
    msg = f"----\|/-\|/-\|/----\n\nTHE WINNER IS...\n  Model: {index} \n {winner}\n\n----\|/-\|/-\|/----\nWe try to do better... with a fine grid search ---->"
    if(stdoutput):
        print(msg)
    if(txt):
        file.write(f"{msg}\n")  

def the_fine_winner_is(file:TextIOWrapper, index:int, true_winner:str, metric:str, stdoutput:bool = True, txt:bool = True):
    msg=f"----\|/-\|/-\|/----\n\nTHE TRUE WINNER IS...\n Model: {index} \n {true_winner} \n with {metric}   <----"
    if(stdoutput):
        print(msg)
    if(txt):
        file.write(f"{msg}\n")   

def model_performance(file:TextIOWrapper, hyperparameters, model_error, stdoutput:bool = True, txt:bool = True):
    msg = f"Model: {hyperparameters} \nMean Train: {model_error['mean_train']} \nMean Mae {model_error['mean_mae']}"\
            f"\nMean Validation: {model_error['mean_validation']}"\
            f"\nMean Rmse: {model_error['mean_rmse']}"\
            f"\nMean MEE: {model_error['mean_mee']}"\
            f"\nMean Epochs: {model_error['mean_epochs']}"\
            f"\n"
    if(model_error.get(f'mean_{C.VALIDATION}_accuracy') !=None):
        msg = f"{msg}Classification Accuracy Training: {model_error[f'mean_{C.TRAINING}_accuracy']} - Validation {model_error[f'mean_{C.VALIDATION}_accuracy']}"
    if(stdoutput):
        print(msg)
    if(txt):
        file.write(f"{msg}\n")

def end_log(file:TextIOWrapper):
    file.close()
     