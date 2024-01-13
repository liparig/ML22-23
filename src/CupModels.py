from cupEvaluation import cup_evaluation, ensemble_Cup
import readMonkAndCup as readMC
import time
import costants as C
#Per avviare la blind con i migliori modelli che abbiamo trovato nelle varie grid search che ne pensi?

#Batch
models=[{C.HYPERPARAMETERS:{'l_dim': [10, 15, 3], 'a_functions': ['tanh', 'identity'], 'eta': 0.004, 'tau': (False, False), 'g_clipping': (False, False), 'dropout': (False, False), 'reg': (False, False), 'dim_batch': 0, 'momentum': ('classic', 0.7), 'epochs': 2000, 'batch_shuffle': True, 'eps': 0.01, 'distribution': 'glorot', 'bias': 0, 'seed': [52], 'classification': False, 'early_stop': True, 'patience': 10, 'treshold_variance': 1e-10}}]
models.append({C.HYPERPARAMETERS:{'l_dim': [10, 15, 3], 'a_functions': ['tanh', 'identity'], 'eta': 0.008, 'tau': (False, False), 'g_clipping': (False, False), 'dropout': (False, False), 'reg': (False, False), 'dim_batch': 0, 'momentum': ('classic', 0.7), 'epochs': 2000, 'batch_shuffle': True, 'eps': 0.01, 'distribution': 'glorot', 'bias': 0, 'seed': [52], 'classification': False, 'early_stop': True, 'patience': 10, 'treshold_variance': 1e-10} } )
models.append({C.HYPERPARAMETERS:{'l_dim': [10, 15, 3], 'a_functions': ['leakyRelu', 'identity'], 'eta': 0.004, 'tau': (False, False), 'g_clipping': (False, False), 'dropout': (False, False), 'reg': (False, False), 'dim_batch': 0, 'momentum': ('classic', 0.7), 'epochs': 2000, 'batch_shuffle': True, 'eps': 0.01, 'distribution': 'glorot', 'bias': 0, 'seed': [52], 'classification': False, 'early_stop': True, 'patience': 10, 'treshold_variance': 1e-10} })
models.append({C.HYPERPARAMETERS: {'l_dim': [10, 8, 5, 3], 'a_functions': ['leakyRelu', 'leakyRelu', 'identity'], 'eta': 0.05, 'tau': (500, 0.004), 'g_clipping': (True, 10), 'dropout': (False, False), 'reg': (False, False), 'dim_batch': 0, 'momentum': ('classic', 0.7), 'epochs': 2000, 'batch_shuffle': True, 'eps': 0.0125, 'distribution': 'glorot', 'bias': 0, 'seed': [52], 'classification': False, 'early_stop': True, 'patience': 10, 'treshold_variance': 1e-10}} )
#DropOut
models.append({C.HYPERPARAMETERS:{'l_dim': [10, 20, 20, 20, 3], 'a_functions': ['leakyRelu', 'leakyRelu', 'leakyRelu', 'identity'], 'eta': 0.01, 'tau': (1000, 0.0004), 'g_clipping': (True, 10), 'dropout': (True, 0.7), 'reg': ('tikhonov', 0.0002), 'dim_batch': 0, 'momentum': ('classic', 0.7), 'epochs': 1000, 'batch_shuffle': True, 'eps': 0.0075, 'distribution': 'glorot', 'bias': 0, 'seed': [52], 'classification': False, 'early_stop': True, 'patience': 15, 'treshold_variance': 1e-08}})

if __name__ == "__main__":

    start = time.time()
    TR_x_CUP, TR_y_CUP,  TS_x_CUP, TS_y_CUP = readMC.get_cup_house_test()
    ensemble_Cup(models, TR_x_CUP, TR_y_CUP, TS_x_CUP, TS_y_CUP, "TestCUP_BestModels", filename = "Best_Ensemble_")

    end = time.time()
    print(f'Cup Test takes {(end - start)/3600} hours')