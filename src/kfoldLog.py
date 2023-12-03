from costants import FORMATTIMESTAMP
import time
import os

      
def start_log(fileName:str):
        # Writing to file
        timestr:str = time.strftime(FORMATTIMESTAMP)
        if(not(os.path.isdir('../KFoldCV'))):
            os.makedirs('../KFoldCV')
        msfile=open(f"../KFoldCV/{fileName}_{timestr}.txt", "w")
        return msfile,timestr

def estimate_model(file,index:int, toal:int, stdoutput:bool=True,txt:bool=True):
    msg=f"---->\nEstimate error for model #{index} of {toal}"
    if(stdoutput):
        print(msg)
    if(txt):
        file.write(msg+"\n")

def the_winner_is(file,index:int, winner:str,  stdoutput:bool=True,txt:bool=True ):
    msg= f"----\|/-\|/-\|/----\n\nTHE WINNER IS...\n  Model: {index} \n {winner}\n\n----\|/-\|/-\|/----\nWe try to do better... with a fine grid search ---->"
    if(stdoutput):
        print(msg)
    if(txt):
        file.write(msg+"\n")

def the_fine_winner_is(file,index:int, true_winner:str, metric:str,  stdoutput:bool=True,txt:bool=True):
    msg=f"----\|/-\|/-\|/----\n\nTHE TRUE WINNER IS...\n Model: {index} \n {true_winner} \n with {metric}   <----"
    if(stdoutput):
        print(msg)
    if(txt):
        file.write(msg+"\n")   

def model_performance(file, hyperparameters,model_error,stdoutput:bool=True,txt:bool=True):
    msg = f"Model: {hyperparameters} \nMean Train: {model_error['mean_train']} \nMean Mae {model_error['mean_mae']}"\
            f"\nMean Validation: {model_error['mean_validation']}"\
            f"\nMean Rmse: {model_error['mean_rmse']}"\
            f"\nMean MEE: {model_error['mean_mee']}"\
            f"\nMean Epochs: {model_error['mean_epochs']}"\
            f"\n"
    if(model_error['mean_v_accuracy']!=None):
        msg = msg + f"Classification Accuracy Training: {model_error['mean_t_accuracy']} - Validation {model_error['mean_v_accuracy']}"
    if(stdoutput):
        print(msg)
    if(txt):
        file.write(msg+"\n")

def end_log(file):
    file.close()
     