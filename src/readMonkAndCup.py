#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import math

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# training and test set - {monk1 - monk2 - monk3}
TRAINMONK1 = BASE_DIR+"/MonkDatasets/monks-1.train"
TRAINMONK2 = BASE_DIR+"/MonkDatasets/monks-2.train"
TRAINMONK3 = BASE_DIR+"/MonkDatasets/monks-3.train"
TRAINCUP = BASE_DIR+"/CupDatasets/Cup23/ML-CUP23-TR.csv"

TESTMONK1 = BASE_DIR+"/MonkDatasets/monks-1.test"
TESTMONK2 = BASE_DIR+"/MonkDatasets/monks-2.test"
TESTMONK3 = BASE_DIR+"/MonkDatasets/monks-3.test"
TESTCUP= BASE_DIR+"/CupDatasets/Cup23/ML-CUP23-TS.csv"

#region Settings for all the reading
def get_train_Monk_1(tanh:bool = False):
    return read_monk(TRAINMONK1, tanh)

def get_test_Monk_1(tanh:bool = False):
    return read_monk(TESTMONK1, tanh)

def get_train_Monk_2(tanh:bool = False):
    return read_monk(TRAINMONK2, tanh)

def get_test_Monk_2(tanh:bool = False):
    return read_monk(TESTMONK2, tanh)

def get_train_Monk_3(tanh:bool = False):
    return read_monk(TRAINMONK3, tanh)

def get_test_Monk_3(tanh:bool = False):
    return read_monk(TESTMONK3, tanh)

def get_test_CUP():
    return read_blind_test_cup(TESTCUP)
#endregion

# Read a generic monk dataset
# :param: path+filename of the dataset file
# :param: tanh is a flag for set if the last activation fuction is a tanh for change the target values
# :return: monk dataset and target values
def read_monk(path:str, tanh:bool = False):
    # read the dataset
    col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    monk_dataset = pd.read_csv(path, sep=' ', names = col_names)
    monk_dataset.set_index('Id', inplace=True)
    
    # shuffle the DataFrame rows
    monk_dataset = monk_dataset.sample(frac = 1)
    
    # get targets from dataset
    monk_targets = monk_dataset.pop('class')
    # 1-hot encoding (and transform dataframe to numpy array)
    monk_dataset = OneHotEncoder().fit_transform(monk_dataset).toarray().astype(np.float64)

    # transform targets from pandas dataframe to numpy ndarray
    monk_targets = monk_targets.to_numpy()[:,np.newaxis].astype(np.float64)
    if tanh:
        monk_targets[monk_targets==0]=-1
    return monk_dataset, monk_targets

# DEPRECATED
def read_monk_Tr_Vl(name:str = TRAINMONK1, perc:float = 0.25):
    # read csv
    col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    monk_dataset = pd.read_csv(name, sep = ' ', names = col_names)
    monk_dataset.set_index('Id', inplace = True)

    # shuffle the DataFrame rows
    monk_dataset = monk_dataset.sample(frac = 1)
    
    # get labels from dataset
    monk_targets = monk_dataset.pop('class').to_numpy(dtype=np.int32)
    
    # 1-hot encoding (and transform dataframe to numpy array)
    monk_dataset = OneHotEncoder().fit_transform(monk_dataset).toarray().astype(np.float64)
    
    dim = math.ceil(len(monk_dataset) * perc)
    
    # inputs = monk_dataset
    # val_inputs = monk_dataset[-dim:,:]
    
    # get targets aside
    inputs = monk_dataset[: -dim,:]
    targets = monk_targets[: -dim]
    val_inputs = monk_dataset[-dim:,:]
    val_targets = monk_targets[-dim:]
    return inputs, targets, val_inputs, val_targets

# Split the training dataset into training and validation
# :param: Tr_x is the training dataset
# :param: Tr_y is the training target
# :param: perc is the percent of the dataset that will become validation dataset
# :return: monk dataset, target values, validation inputs and validation targets
def split_Tr_Val(Tr_x:list, Tr_y:list, perc:float = 0.25):
    dim = math.ceil(len(Tr_x) * perc)
    
    # get targets aside
    inputs = Tr_x[: -dim,:]
    targets = Tr_y[: -dim]
    val_inputs = Tr_x[-dim:,:]
    val_targets = Tr_y[-dim:]
    
    return inputs, targets, val_inputs, val_targets

def read_cup(name):
    # get directory
    targets = []
    # read csv
    col_names = ['Id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'target_x', 'target_y','target_z']

    cup_dataset = pd.read_csv(name, sep=',', skiprows=range(7), names=col_names)

    cup_dataset.set_index('Id', inplace=True)
    # shuffle the DataFrame rows
    cup_dataset = cup_dataset.sample(frac = 1)
    if name == TRAINCUP:
        # get targets aside
        target_x = cup_dataset.pop('target_x').to_numpy(dtype=np.float64)
        target_y = cup_dataset.pop('target_y').to_numpy(dtype=np.float64)
        target_z = cup_dataset.pop('target_z').to_numpy(dtype=np.float64)
        targets = np.vstack((target_x,target_y,target_z)).T
    
    inputs = cup_dataset.to_numpy(dtype=np.float64)
    # transform labels from pandas dataframe to numpy ndarray
    
    return inputs, targets

# Read cup dataset for test
# :param: perc is the percent of the dataset that will become test dataset
# :return: training dataset, training values, test inputs and test targets
def get_cup_house_test(perc:float = 0.25):
    
    # get directory
    targets=[]
    # read csv
    col_names = ['Id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'target_x', 'target_y','target_z']
    cup_dataset = pd.read_csv(TRAINCUP, sep=',', skiprows = range(7), names = col_names)
    
    cup_dataset.set_index('Id', inplace=True)
    # shuffle the DataFrame rows
    
    cup_dataset = cup_dataset.sample(frac = 1)
    dim = math.ceil(len(cup_dataset) * perc)
    inputs = cup_dataset.to_numpy(dtype=np.float64)[: -dim,:]
    test_inputs = cup_dataset.to_numpy(dtype=np.float64)[-dim:,:]
    
    # get targets aside
    inputs, targets = inputs[:, :-3], inputs[:, -3:]
    test_inputs, test_targets=test_inputs[:, :-3], test_inputs[:, -3:]
    
    # transform labels from pandas dataframe to numpy ndarray
    
    return inputs, targets, test_inputs, test_targets

def read_blind_test_cup(name):
    # get directory
    # targets = []
    # read csv
    col_names = ['Id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9','a10']

    cup_dataset = pd.read_csv(name, sep=',', skiprows=range(7), names=col_names)

    cup_dataset.set_index('Id', inplace=True)
    
    inputs = cup_dataset.to_numpy(dtype=np.float64)
    # transform labels from pandas dataframe to numpy ndarray
    
    return inputs