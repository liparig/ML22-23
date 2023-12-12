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

def get_train_Monk_1():
    return read_monk(TRAINMONK1)

def get_test_Monk_1():
    return read_monk(TESTMONK1)

def get_train_Monk_2():
    return read_monk(TRAINMONK2)

def get_test_Monk_2():
    return read_monk(TESTMONK2)

def get_train_Monk_3():
    return read_monk(TRAINMONK3)

def get_test_Monk_3():
    return read_monk(TESTMONK3)

def get_train_CUP():
    return read_cup(TRAINCUP)

def get_test_CUP():
    return read_cup(TESTCUP)

def read_monk(name):
    # read the dataset
    col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    monk_dataset = pd.read_csv(name, sep=' ', names=col_names)
    monk_dataset.set_index('Id', inplace=True)
    
    # shuffle the DataFrame rows
    monk_dataset = monk_dataset.sample(frac = 1)
    
    # get labels from dataset
    monk_targets = monk_dataset.pop('class')
    
    # 1-hot encoding (and transform dataframe to numpy array)
    monk_dataset = OneHotEncoder().fit_transform(monk_dataset).toarray().astype(np.int64)
    
    # transform labels from pandas dataframe to numpy ndarray
    monk_targets = monk_targets.to_numpy()[:, np.newaxis]

    return monk_dataset, monk_targets

def read_monk_Tr_Vl(name:str = TRAINMONK1, perc:float = 0.25):
    # read csv
    col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    monk_dataset = pd.read_csv(name, sep = ' ', names = col_names)
    monk_dataset.set_index('Id', inplace = True)

    # shuffle the DataFrame rows
    monk_dataset = monk_dataset.sample(frac = 1)
    
    # get labels from dataset
    monk_targets = monk_dataset.pop('class').to_numpy(dtype=np.float32)
    
    # 1-hot encoding (and transform dataframe to numpy array)
    monk_dataset = OneHotEncoder().fit_transform(monk_dataset).toarray().astype(np.int64)
    
    dim = math.ceil(len(monk_dataset) * perc)
    
    # inputs = monk_dataset
    # val_inputs = monk_dataset[-dim:,:]
    
    # get targets aside
    inputs = monk_dataset[: -dim,:]
    targets = monk_targets[: -dim]
    val_inputs = monk_dataset[-dim:,:]
    val_targets = monk_targets[-dim:]
    
    return inputs, targets, val_inputs, val_targets

def read_cup(name,targetCol=3):
    # get directory
    targets=[]
    # read csv
    col_names = ['Id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'target_x', 'target_y','target_z']

  
        
    cup_dataset = pd.read_csv(name, sep=',', skiprows=range(7), names=col_names)

    cup_dataset.set_index('Id', inplace=True)
    # shuffle the DataFrame rows
    cup_dataset = cup_dataset.sample(frac = 1)
    print(cup_dataset)
    if name == TRAINCUP:
        # get targets aside
        target_x = cup_dataset.pop('target_x').to_numpy(dtype=np.float32)
        target_y = cup_dataset.pop('target_y').to_numpy(dtype=np.float32)
        target_z = cup_dataset.pop('target_z').to_numpy(dtype=np.float32)
        targets=np.vstack((target_x,target_y,target_z)).T
    
    inputs = cup_dataset.to_numpy(dtype=np.float32)
    # transform labels from pandas dataframe to numpy ndarray
    
    return inputs, targets

def get_cup_house_test(perc=0.25):
    
    # get directory
    targets=[]
    # read csv
    col_names = ['Id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'target_x', 'target_y','target_z']
    cup_dataset = pd.read_csv(TRAINCUP, sep=',', skiprows = range(7), names = col_names)
    
    cup_dataset.set_index('Id', inplace=True)
    # shuffle the DataFrame rows
    
    cup_dataset = cup_dataset.sample(frac = 1)
    dim = math.ceil(len(cup_dataset) * perc)
    inputs = cup_dataset.to_numpy(dtype=np.float32)[: -dim,:]
    val_inputs =cup_dataset.to_numpy(dtype=np.float32)[-dim:,:]
    
    # get targets aside
    inputs,targets = inputs[:, :-2], inputs[:, -2:]
    val_inputs,val_targets=val_inputs[:, :-2], val_inputs[:, -2:]
    
    # transform labels from pandas dataframe to numpy ndarray
    
    return inputs, targets,val_inputs,val_targets