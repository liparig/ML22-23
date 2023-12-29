from io import TextIOWrapper
import costants as C
import time
import os
import numpy as np
def Model_Assessment_log(fileDir:str, fileName:str, hyperparameters:str, result:str):
        # Writing to file
        timestr:str = time.strftime(C.FORMATTIMESTAMP)
        if(not(os.path.isdir(C.PATH_MODEL_ASSESSMENT_DIR))):
            os.makedirs(C.PATH_MODEL_ASSESSMENT_DIR)
        if(not(os.path.isdir(f"{C.PATH_MODEL_ASSESSMENT_DIR}/{fileDir}"))):
            os.makedirs(f"{C.PATH_MODEL_ASSESSMENT_DIR}/{fileDir}")
        mafile = open(f"{C.PATH_MODEL_ASSESSMENT_DIR}/{fileDir}/{fileName}_{timestr}.txt", "a")
        mafile.write(f"{hyperparameters}\n")
        mafile.write(f"{result}\n") 
        
        mafile.close() 
        return mafile, timestr

def Model_Assessment_Outputs(results,fileDir:str,fileName,timestamp):# Nomi delle colonne
    col_names = ['target_y', 'target_x', 'target_z', 'out_y', 'out_x', 'out_z']
    # Aggiungi i nomi delle colonne come prima riga
    result_with_header = np.vstack([col_names, results])
    np.savetxt(f'{C.PATH_MODEL_ASSESSMENT_DIR}/{fileDir}/{fileName}_output_{timestamp}.csv', result_with_header, delimiter=',', fmt='%s', header='', comments='')  

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
     