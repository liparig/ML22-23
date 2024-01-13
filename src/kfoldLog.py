from io import TextIOWrapper
import costants as C
import time
import os
import numpy as np
from functools import wraps

# It's an annotation for write the timing of the execution function
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        timesDir:str = '../times'
        if(not(os.path.isdir(timesDir))):
            os.makedirs(timesDir)
        with open(os.path.join(f'{timesDir}',f"time_{func.__name__}.txt"), "a") as w:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            #print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
            w.write(f"Function {func.__name__} Took {total_time:.4f} seconds\n")
            if(func.__name__ == 'estimate_model_error'):
                for arg in args:             
                    w.write(f"{arg}\n")
                for key, value in kwargs.items(): 
                    w.write(f"{key} == {value}\n")
        return result

    return timeit_wrapper

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
        with open(f"{C.PATH_MODEL_ASSESSMENT_DIR}/{fileDir}/{fileName}_{timestr}.txt", "a") as mafile:
                mafile.write(f"{hyperparameters}\n")
                mafile.write(f"{result}\n") 

        return mafile, timestr

# Make a file csv with the output and the target of the test
# :param: the result from the test
# :param: fileDir is the directory of the file
# :param: fileName is the name of the file
# :param: timestamp 
def Model_Assessment_Outputs(results, fileDir:str, fileName:str,col_names = None, timestamp=False):
        timestamp_str = time.strftime(C.FORMATTIMESTAMP) if timestamp else ""
        if(not(os.path.isdir(f'{C.PATH_MODEL_ASSESSMENT_DIR}/{fileDir}/'))):
            os.makedirs(f'{C.PATH_MODEL_ASSESSMENT_DIR}/{fileDir}/')
        if col_names is None:
            col_names = ['target_y', 'target_x', 'target_z', 'out_y', 'out_x', 'out_z']

        # Save the array with the correct format specifier
        np.savetxt(
            f'{C.PATH_MODEL_ASSESSMENT_DIR}/{fileDir}/{fileName}_output_{timestamp_str}.csv',
            results,
            delimiter=',',
            fmt='%s',
            header=','.join(col_names),
            comments=''
        )
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
        listId: list[int] = [int(i) for i in range(1, results.shape[0]+1)]
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
        # Specify the directory path
        directory_path = "../times"

        # Specify the string you want to append to each file
        string_to_append = "=================================================\n"

        # Iterate over files in the directory
        for filename in os.listdir(directory_path):
            # Check if the item is a file (not a directory)
            if os.path.isfile(os.path.join(directory_path, filename)):
                # Open the file in append mode and write the string
                with open(os.path.join(directory_path, filename), 'a') as file:
                    file.write(string_to_append + '\n')  # Add a newline if needed

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
    if(model_error.get(f'mean_{C.VALIDATION}_{C.ACCURACY}') !=None):
        msg = f"{msg}Classification Accuracy Training: {model_error[f'mean_{C.TRAINING}_{C.ACCURACY}']} - Validation {model_error[f'mean_{C.VALIDATION}_{C.ACCURACY}']}"
    if(stdoutput):
        print(msg)
    if(txt):
        file.write(f"{msg}\n")

# Close the file of the Logger
# :param: file is the object for write and read in the logger file
def end_log(file:TextIOWrapper):
    file.close()
     