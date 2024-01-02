from io import TextIOWrapper
import costants as C
import time
import os
import numpy as np

# Write on a file the model assesment
# :param: fileDir is the directory of the file
# :param: fileName is the name of the file
# :param: HyperParameters are the configuration parameters
# :param: result is the value computed from the phase
# :return: the object file and the timestamp
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

# Make a file csv with the output and the target of the test
# :param: the result from the test
# :param: fileDir is the directory of the file
# :param: fileName is the name of the file
# :param: timestamp 
def Model_Assessment_Outputs(results, fileDir:str, fileName:str, timestamp):
    # name of the column
    col_names = ['target_y', 'target_x', 'target_z', 'out_y', 'out_x', 'out_z']
    # add the name of the column in the first row
    result_with_header = np.vstack([col_names, results])
    np.savetxt(f'{C.PATH_MODEL_ASSESSMENT_DIR}/{fileDir}/{fileName}_output_{timestamp}.csv', result_with_header, delimiter=',', fmt='%s', header='', comments='')  

# Make a file csv for the template for the Cup
# :param: the result from the test
# :param: fileDir is the directory of the file
# :param: fileName is the name of the file
# :param: timestamp 
def ML_Cup_Template(results, fileDir:str, fileName:str, timestamp=False):
   # name of the column
    # Name1 Surname2, Name2 Surname2, Name3 Surname3 
    # Team Name
    # ML-CUP23
    if not timestamp:
        timestamp:str = time.strftime(C.FORMATTIMESTAMP)
    header = np.array(['# Giuseppe Lipari', ' Carmine Vitiello','',''])
    header2 = np.array(['# Team Name ','','',''])
    header3 = np.array(['# ML-CUP23 ','','',''])
    col_names = np.array(['# id', ' out_y', ' out_x', ' out_z'])
    # add the name of the column in the first row
    listId:list[int] = []
    for i in range(1, results.shape[0]+1):
        listId.append(int(i))
    results = np.concatenate((np.array(listId).astype(int).reshape((results.shape[0],1)), results), axis=1)
    # print(header.shape, col_names.shape, results.shape)
    result_with_header = np.vstack([header, header2, header3, col_names, results])
    np.savetxt(f'{C.PATH_MODEL_ASSESSMENT_DIR}/{fileDir}/{fileName}_output_{timestamp}.csv', result_with_header, delimiter=',', fmt='%s', header='', comments='')  

# Initialitation of the logger
# :param: fileName is the name of the file
# :return: the object file and the timestamp
def start_log(fileName:str):
        # Writing to file
        timestr:str = time.strftime(C.FORMATTIMESTAMP)
        if(not(os.path.isdir(C.PATH_KFOLDCV_DIR))):
            os.makedirs(C.PATH_KFOLDCV_DIR)
        msfile = open(f"{C.PATH_KFOLDCV_DIR}/{fileName}_{timestr}.txt", "w")
        return msfile, timestr

# Logger in estimate of the models
# :param: file is the object for write and read in the logger file 
# :param: index on total number of the models
# :param: total is the number of models
# :param: stdoutput is a flag for write in the shell output
# :param: txt is a flag for write in the logger file
def estimate_model(file:TextIOWrapper, index:int, total:int, stdoutput:bool = True, txt:bool = True):
    msg = f"---->\nEstimate error for model #{index} of {total}"
    if(stdoutput):
        print(msg)
    if(txt):
        file.write(f"{msg}\n")  

# Logger the coarse winner model
# :param: file is the object for write and read in the logger file 
# :param: index on total number of the models
# :param: winner is the configuration parameters of the winner model
# :param: stdoutput is a flag for write in the shell output
# :param: txt is a flag for write in the logger file
def the_winner_is(file:TextIOWrapper, index:int, winner:str, stdoutput:bool = True, txt:bool = True):
    msg = f"----\|/-\|/-\|/----\n\nTHE WINNER IS...\n  Model: {index} \n {winner}\n\n----\|/-\|/-\|/----\nWe try to do better... with a fine grid search ---->"
    if(stdoutput):
        print(msg)
    if(txt):
        file.write(f"{msg}\n")  

# Logger the fine winner model
# :param: file is the object for write and read in the logger file 
# :param: index on total number of the models
# :param: true_winner is the configuration parameters of the winner model
# :param: metric is value of the metric for evaluation of the winner model
# :param: stdoutput is a flag for write in the shell output
# :param: txt is a flag for write in the logger file
def the_fine_winner_is(file:TextIOWrapper, index:int, true_winner:str, metric:str, stdoutput:bool = True, txt:bool = True):
    msg=f"----\|/-\|/-\|/----\n\nTHE TRUE WINNER IS...\n Model: {index} \n {true_winner} \n with {metric}   <----"
    if(stdoutput):
        print(msg)
    if(txt):
        file.write(f"{msg}\n")   

# Logger the model performance statistics
# :param: file is the object for write and read in the logger file 
# :param: hyperparameters
# :param: model_error is the object with all the mean errors
# :param: stdoutput is a flag for write in the shell output
# :param: txt is a flag for write in the logger file
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

# Close the file of the Logger
# :param: file is the object for write and read in the logger file
def end_log(file:TextIOWrapper):
    file.close()
     