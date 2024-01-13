import time
import costants as C
from cupEvaluation import cup_evaluation, ensemble_Cup
import readMonkAndCup as readMC

#it's a test for get the performance of the best model selected

def main(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, dirName):
    batch1 = {
        'l_dim': [[10, 15, 3]],
        'a_functions': [['tanh', 'identity']],
        'eta': [0.004],
        'tau': [[False, False]],
        'g_clipping': [[False, False]],
        'dropout': [[False, False]],
        'reg': [[False, False]],
        'dim_batch': [0],
        'momentum': [['classic', 0.7]],
        'epochs': [2000],
        'batch_shuffle': [True],
        'eps': [0.01],
        'distribution': ['glorot'],
        'bias': [0],
        'seed': [52],
        'classification': False,
        'early_stop': True,
        'patience': [10],
        'treshold_variance': [1e-10],
    }

    batch2 = {
        'l_dim': [[10, 15, 3]],
        'a_functions': [['tanh', 'identity']],
        'eta': [0.008],
        'tau': [[False, False]],
        'g_clipping': [[False, False]],
        'dropout': [[False, False]],
        'reg': [[False, False]],
        'dim_batch': [0],
        'momentum': [['classic', 0.7]],
        'epochs': [2000],
        'batch_shuffle': [True],
        'eps': [0.01],
        'distribution': ['glorot'],
        'bias': [0],
        'seed': [52],
        'classification': False,
        'early_stop': True,
        'patience': [10],
        'treshold_variance': [1e-10],
    }

    batch3 = {
        'l_dim': [[10, 15, 3]],
        'a_functions': [['leakyRelu', 'identity']],
        'eta': [0.004],
        'tau': [[False, False]],
        'g_clipping': [[False, False]],
        'dropout': [[False, False]],
        'reg': [[False, False]],
        'dim_batch': [0],
        'momentum': [['classic', 0.7]],
        'epochs': [2000],
        'batch_shuffle': [True],
        'eps': [0.01],
        'distribution': ['glorot'],
        'bias': [0],
        'seed': [52],
        'classification': False,
        'early_stop': True,
        'patience': [10],
        'treshold_variance': [1e-10],
    }

    batch4 = {
        'l_dim': [[10, 15, 8, 3]],
        'a_functions': [['leakyRelu', 'leakyRelu', 'identity']],
        'eta': [0.05],
        'tau': [[500, 0.004]],
        'g_clipping': [[True, 7]],
        'dropout': [[False, False]],
        'reg': [[False, False]],
        'dim_batch': [0],
        'momentum': [['classic', 0.7]],
        'epochs': [2000],
        'batch_shuffle': [True],
        'eps': [0.0125],
        'distribution': ['glorot'],
        'bias': [0],
        'seed': [52],
        'classification': False,
        'early_stop': True,
        'patience': [10],
        'treshold_variance': [1e-10],
    }

    dropout1 = {
        'l_dim': [[10, 20, 20, 20, 3]],
        'a_functions': [['leakyRelu', 'leakyRelu', 'leakyRelu', 'identity']],
        'eta': [0.004],
        'tau': [[1000, 0.0004]],
        'g_clipping': [[True, 10]],
        'dropout': [[True, 0.7]],
        'reg': [['tikhonov', 0.0002]],
        'dim_batch': [200],
        'momentum': [['classic', 0.7]],
        'epochs': [1000],
        'batch_shuffle': [True],
        'eps': [0.0075],
        'distribution': ['glorot'],
        'bias': [0],
        'seed': [52],
        'classification': False,
        'early_stop': True,
        'patience': [15],
        'treshold_variance': [1e-08],
    }

    minibatch1 = {
        'l_dim': [[10, 15, 3]],
        'a_functions': [['leakyRelu', 'identity']],
        'eta': [0.005],
        'tau': [[500, 0.0005]],
        'g_clipping': [[True, 10]],
        'dropout': [[False, False]],
        'reg': [[False, False]],
        'dim_batch': [150],
        'momentum': [['classic', 0.7]],
        'epochs': [500],
        'batch_shuffle': [True],
        'eps': [0.001],
        'distribution': ['glorot'],
        'bias': [0],
        'seed': [52],
        'classification': False,
        'early_stop': True,
        'patience': [10],
        'treshold_variance': [1e-08],
    }

    minibatch2 = {
        'l_dim': [[10, 15, 3]],
        'a_functions': [['leakyRelu', 'identity']],
        'eta': [0.00375],
        'tau': [[500, 0.0005]],
        'g_clipping': [[True, 10]],
        'dropout': [[False, False]],
        'reg': [[False, False]],
        'dim_batch': [100],
        'momentum': [['classic', 0.7]],
        'epochs': [500],
        'batch_shuffle': [True],
        'eps': [0.001],
        'distribution': ['glorot'],
        'bias': [0],
        'seed': [52],
        'classification': False,
        'early_stop': True,
        'patience': [10],
        'treshold_variance': [1e-08],
    }   
    models = []
    best = cup_evaluation(inTR_x_cup,inTR_y_cup,inTS_x_cup,inTS_y_cup,batch1,dirName=dirName,prefixFilename="Batch1_",fold=5,FineGS=False)
    models.extend(best)

    best = cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, batch2, dirName, prefixFilename = "Batch2_", fold = 5,FineGS=False)
    models.extend(best)
    best = cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, batch3, dirName, prefixFilename = "Batch3_", fold = 5,FineGS=False)
    models.extend(best)

    best = cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, batch4, dirName, prefixFilename = "Batch4_", fold = 5,FineGS=False)
    models.extend(best)

    best = cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, dropout1, dirName, prefixFilename = "Dropout1_", fold = 5,FineGS=False)
    models.extend(best)

    best = cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, minibatch1, dirName, prefixFilename = "MiniBatch1_", fold = 5,FineGS=False)
    models.extend(best)

    best = cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, minibatch2, dirName, prefixFilename = "MiniBatch2_", fold = 5,FineGS=False)
    models.extend(best)

    ensemble_Cup(models, inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, dirName, filename = "BestModelesEnsemble_")
    all_tr_x_cup,all_tr_y_cup,_,_= readMC.get_cup_house_test(perc=0)
    ensemble_Cup(models,all_tr_x_cup,all_tr_y_cup,dirName=dirName,filename="BlindEnsemble_")
if __name__ == "__main__":
    start = time.time()
    TR_x_CUP, TR_y_CUP,  TS_x_CUP, TS_y_CUP = readMC.get_cup_house_test()
    DIRNAME:str = "TestBestModels"
    main(TR_x_CUP, TR_y_CUP, TS_x_CUP, TS_y_CUP, DIRNAME)
    end = time.time()

    print(f'Cup Test takes {(end - start)/3600} hours')