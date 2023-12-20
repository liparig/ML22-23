import numpy as np

import costants as C

#classe che raggruppa gli hyperparametri in un oggetto Candidate
class Candidate:
    def __init__(self,candidate):    
        self.l_dim = candidate['l_dim']
        self.a_functions = candidate['a_functions']
        self.eta = candidate['eta']
        self.tau = candidate['tau']
        self.reg = candidate['reg']
        self.dim_batch = candidate['dim_batch']
        self.momentum = candidate['momentum']
        self.epochs = candidate['epochs']
        self.batch_shuffle = candidate['batch_shuffle']
        self.eps = candidate['eps']
        self.distribution = candidate['distribution']
        self.bias = candidate['bias']
        self.seed = candidate['seed']
        self.classification = candidate['classification']
        self.early_stop = candidate['early_stop']
        self.patience = candidate['patience']
        self.treshold_variance = candidate['treshold_variance']

    def get_fine_range(self,value):
        """
        :param param: a specific value for an hyperparameter
        :return: a small range of this values
        """
        higher = 1.25
        lower = 0.75
        return [np.round(value * lower, 7),
                value,
                np.round(value * higher, 7)]

    def get_fine_batch_size(self,value):
        """
        :param param: a specific value for the hyperparameter "batch size"
        :return: a small range of this value
        """
        if value == 0:
            return [0]
        else:
            return [max(1, value - 10),
                value,
                int(value + 10)]
        
    def get_fine_int_range(self,value):
        """
        :param param: a specific value for an hyperparameter
        :return: a small range of this values
        """
        higher = 1.25
        lower = 0.75
        return [int(value * lower),
                value,
                int(value * higher)]


    def get_fine_tuple(self,value):
        if not isinstance(value[1], int) or not isinstance(value[1], float):
            return [value]
        higher = 1.25
        lower = 0.8
        return [(value[0],np.round(value * lower, 7)),
                value,
                (value[0],np.round(value * higher, 7))]

    def to_string(self):
        return f" Hyperparameters: 'l_dim':{self.l_dim},\
            'a_functions':{self.a_functions},\
            'eta':{self.eta},\
            'tau': {self.tau},\
            'reg':{self.reg},\
            'dim_batch':{self.dim_batch},\
            'momentum': {self.momentum},\
            'epochs':{self.epochs},\
            'batch_shuffle':{self.batch_shuffle},\
            'eps': {self.eps},\
            'distribution':{self.distribution},\
            'bias':{self.bias},\
            'seed': {self.seed},\
            'classification':{self.classification},\
            'early_stop':{self.early_stop},\
            'patience': {self.patience}"
            #'treshold_variance':{self.treshold_variance}"

    def get_dictionary(self):
        return {'l_dim':self.l_dim,
            'a_functions':self.a_functions,
            'eta':self.eta,
            'tau': self.tau,
            'reg':self.reg,
            'dim_batch':self.dim_batch,
            'momentum': self.momentum,
            'epochs':self.epochs,
            'batch_shuffle':self.batch_shuffle,
            'eps': self.eps,
            'distribution':self.distribution,
            'bias':self.bias,
            'seed': self.seed,
            'classification':self.classification,
            'early_stop':self.early_stop,
            'patience': self.patience,
            'treshold_variance':self.treshold_variance
            }

class CandidatesHyperparameters:
   
    def __init__(self):
        self.l_dim = []
        self.a_functions = []
        self.eta = []
        self.tau = []
        self.reg = []
        self.dim_batch = []
        self.momentum = []
        self.epochs = []
        self.batch_shuffle = []
        self.eps = []
        self.distribution = []
        self.bias = []
        self.seed = []
        self.classification = []
        self.early_stop = []
        self.patience = []
        self.treshold_variance = []
        self.count = 0

    def insert_candidate(self, l_dim, a_functions=C.AFUNCTION, eta=C.ETA, tau=C.TAU, reg=C.REG,\
        dim_batch=C.BATCH, epochs=C.EPOCHS,batch_shuffle=True,momentum=C.MOMENTUM,eps=C.EPS,distribution=C.UNIFORM,\
        bias=C.BIAS,seed=C.R_SEEDS, early_stop=True, classification=False,patience=C.PATIENCE,treshold_variance=C.TRESHOLDVARIANCE):
        self.l_dim.append(l_dim)
        self.a_functions.append(a_functions)
        self.eta.append(eta)
        self.tau.append(tau)
        self.reg.append(reg)
        self.dim_batch.append(dim_batch)
        self.momentum.append(momentum)
        self.epochs.append(epochs)
        self.batch_shuffle.append(batch_shuffle)
        self.eps.append(eps)
        self.distribution.append(distribution)
        self.bias.append(bias)
        self.seed.append(seed)
        self.classification.append(classification)
        self.early_stop.append(early_stop)
        self.patience.append(patience)
        self.treshold_variance.append(treshold_variance)
        self.count+=1

    def get_candidate_dict(self,index):
        return {'l_dim':self.l_dim[index],
            'a_functions':self.a_functions[index],
            'eta':self.eta[index],
            'tau': self.tau[index],
            'reg':self.reg[index],
            'dim_batch':self.dim_batch[index],
            'momentum': self.momentum[index],
            'epochs':self.epochs[index],
            'batch_shuffle':self.batch_shuffle[index],
            'eps': self.eps[index],
            'distribution':self.distribution[index],
            'bias':self.bias[index],
            'seed': self.seed[index],
            'classification':self.classification[index],
            'early_stop':self.early_stop[index],
            'patience': self.patience[index],
            'treshold_variance':self.treshold_variance[index],
            }
    def get_all_candidates_dict(self):
        candidates:list = []
        for index in range (self.count):
            candidates.append(self.get_candidate_dict(index))
        return candidates
    
    def set_project_hyperparameters(self, namedataset):
        """
        :param dataset: quale dataset utilizzare
        :return: set di valori per gli hyperparameters per la Model Selection
        """
        if namedataset == 'monk':
            self.l_dim = [[17,2,1]]
            self.a_functions = [[C.RELU, C.TANH]]#,[C.RELU, C.SIGMOID]
            self.eta = [0.4]
            self.momentum = [(C.NESTEROV, 0.5)]
            self.reg = [ (False, False)] #(TIKHONOV, 0.01), (LASSO, 0.01), (False, False)
            self.dim_batch = [0,15]
            self.tau = [(1000,0.04)]
            self.patience = [100]
            self.early_stop = True
            self.eps = [0.05, 0.1]
            self.distribution = [C.UNIFORM]
            self.bias = [0]
            self.epochs = [1000]
            self.classification=True
            self.treshold_variance = [1.e-6]
            # self.l_dim = [[17, 12, 6, 1], [17, 5, 5, 1]]
            # self.a_functions = [[SIGMOID, SIGMOID, TANH], [RELU, RELU, SIGMOID]]
            # self.eta = [0.5]
            # self.momentum = [('classic', 0.5)]
            # self.reg = [(False,False), ('tikhonov', 0.01), ('lasso', 0.01)] #('tikhonov', 0.001), ('lasso', 0.001),
            # self.dim_batch = [1, 50]
            # self.tau = [(False,False), (30, 0.005)]
            # self.patience = [50]
            # self.eps=[0.5]
            # self.distribution=['uniform']
            # self.bias = [0]
            # self.classification=[True]
            '''
            self.l_dim = [[17,10,1],[17,4,1],[17,8,1]]
            self.a_functions = [[SIGMOID, TANH],[RELU, SIGMOID]]
            self.eta=[0.5, 0.2,0.01]
            self.momentum=[ ('',0) , ('nesterov',0.75) , ('nesterov',0.5) , ('classic',0.75), ('classic',0.5) ]
            self.reg=[ (False,False),  ('tikhonov',0.001), ('lasso',0.001),('tikhonov',0.0001),  ('lasso',0.0001) ]
            self.dim_batch=[0,1,50]
            self.tau=[ (False,False), (25,0.005),  (25,0.0005) ]
            self.patience=[50]
            self.eps=[0.7,0.5]
            self.distribution=['uniform']
            self.bias = [0]
            self.classification=[True]
            '''
        elif namedataset=="cup":
            self.l_dim = [[9,16,8,3],[9,32,16,3],[9,64,32,16,3]]
            self.a_functions = [[C.SIGMOID,C.SIGMOID,'identity'],['relu','relu','identity'],['relu','relu','relu','identity'],[C.SIGMOID,C.SIGMOID,C.SIGMOID,'identity']]
            self.eta=[0.1, 0.01]
            self.momentum=[ ('',0) , ('nesterov',0.5) , ('classic',0.5) ]
            self.reg=[ (False,False), ('tikhonov',0.0001),  ('lasso',0.0001) ]
            self.dim_batch=[0,25]
            self.tau=[ (False,False), (100,0.005), (70,0.001)]
            self.patience=[20]
            self.eps=[0.1,0.3]
            self.distribution=['uniform']
            self.bias = [0]
            self.classification=False
        
   
                
