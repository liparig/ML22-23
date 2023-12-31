import numpy as np

import costants as C

# Candidate is the class for groupping the hyperparameters in an object
# it's an utility class for manage the candidates 
class Candidate:
    def __init__(self,candidate):    
        self.l_dim = candidate[C.L_NET]
        self.a_functions = candidate[C.L_ACTIVATION]
        self.eta = candidate[C.L_ETA]
        self.tau = candidate[C.L_TAU]
        self.g_clipping = candidate[C.L_G_CLIPPING]
        self.reg = candidate[C.L_REG]
        self.dim_batch = candidate[C.L_DIMBATCH]
        self.momentum = candidate[C.L_MOMENTUM]
        self.epochs = candidate[C.L_EPOCHS]
        self.batch_shuffle = candidate[C.L_SHUFFLE]
        self.eps = candidate[C.L_EPS]
        self.distribution = candidate[C.L_DISTRIBUTION]
        self.bias = candidate[C.L_BIAS]
        self.seed = candidate[C.L_SEED]
        self.classification = candidate[C.L_CLASSIFICATION]
        self.early_stop = candidate[C.L_EARLYSTOP]
        self.patience = candidate[C.L_PATIENCE]
        self.treshold_variance = candidate[C.L_TRESHOLD_VARIANCE]

    # computes a fine range near to a single value
    # :param: a specific value for an hyperparameter
    # :return: a small range of this values
    def get_fine_range(self, value):
        higher = 1.25
        lower = 0.75
        return [np.round(value * lower, 7),
                value,
                np.round(value * higher, 7)]

    # computes a fine range near to a single value for the batch size
    # :param: a specific value for the hyperparameter "batch size"
    # :return: a small range of this value
    def get_fine_batch_size(self, value):
        higher = 1.10
        lower = 0.90
        if value == 0:
            return [0]
        else:
            return [
                max(1, int(np.round(value * lower))),
                value,
                int(np.round(value * higher))
            ]
        
    # computes a fine range near to an integer single value
    # :param: a specific value for an hyperparameter
    # :return: a small range of this integer values
    def get_fine_int_range(self, value):
        higher:float = 1.25
        lower:float = 0.75
        return [
            int(value * lower),
            value,
            int(value * higher)
        ]

    # computes a fine range near to a tuple single value
    # :param: a specific value for an hyperparameter
    # :return: a small range of this tuple values
    def get_fine_tuple(self, value):
        if not isinstance(value[1], int) or not isinstance(value[1], float):
            return [value]
        higher = 1.25
        lower = 0.8
        return [(value[0],np.round(value * lower, 7)),
                value,
                (value[0],np.round(value * higher, 7))]

    # :return: the string with all the hyperparameters of a configuration theta
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
            'patience': {self.patience}, \
            'treshold_variance':{self.treshold_variance}"

    # :return: the dictionary object with all the hyperparameters of a configuration theta
    def get_dictionary(self):
        return {C.L_NET:self.l_dim,
            C.L_ACTIVATION:self.a_functions,
            C.L_ETA:self.eta,
            C.L_TAU: self.tau,
            C.L_REG:self.reg,
            C.L_DIMBATCH:self.dim_batch,
            C.L_MOMENTUM: self.momentum,
            C.L_EPOCHS:self.epochs,
            C.L_SHUFFLE:self.batch_shuffle,
            C.L_EPS: self.eps,
            C.L_DISTRIBUTION:self.distribution,
            C.L_BIAS:self.bias,
            C.L_SEED: self.seed,
            C.L_CLASSIFICATION:self.classification,
            C.L_EARLYSTOP:self.early_stop,
            C.L_PATIENCE: self.patience,
            C.L_TRESHOLD_VARIANCE:self.treshold_variance
        }
        

class CandidatesHyperparameters:
   
    def __init__(self):
        self.l_dim = []
        self.a_functions = []
        self.eta = []
        self.tau = []
        self.g_clipping = []
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

    # :return: true if there are not candidates else returns false
    def empty(self):
        return self.count == 0
    
    # add a candidatie to the list of candidates
    # :param: l_dim is the number of the units of a layer of the network
    # :param: a_functions is the list of the activation function 
    # :param: eta is the grade of learning
    # :param: tau is the couple of lambda decay (number of the epochs, tau value)
    # :param: reg is the couple (type Regularitation, lambda value)
    # :param: dim_batch is the dimension of the batch
    # :param: epochs is the max number of the epochs
    # :param: batch_shuffle is true if shuffle is enabled
    # :param: momentum is the couple (type momentum, alpha value)
    # :param: eps is the value for the initialitation of the weights
    # :param: distribution is for the generation of the starting weights
    # :param: bias is for initial weight w0 
    # :param: seed is for the generator of the initialitation of the weights
    # :param: early_stop is the flag of the early stop
    # :param: classification is the flag if it is a classification problem
    # :param: patience is the max number of epochs with small or null changes 
    # :param: treshold_variance is the threshold for the patience and early stop
    def insert_candidate(self, l_dim, a_functions=C.AFUNCTION, eta=C.ETA, tau=C.TAU, g_clipping=C.G_CLIPPING, reg=C.REG,\
        dim_batch=C.BATCH, epochs=C.EPOCHS,batch_shuffle=C.BATCH_SHUFFLE,momentum=C.MOMENTUM,eps=C.EPS,distribution=C.UNIFORM,\
        bias=C.BIAS,seed=C.R_SEEDS, early_stop=True, classification=False,patience=C.PATIENCE,treshold_variance=C.TRESHOLDVARIANCE):
        self.l_dim.append(l_dim)
        self.a_functions.append(a_functions)
        self.eta.append(eta)
        self.tau.append(tau)
        self.g_clipping.append(g_clipping)
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

    # :param: index of a single candidate object
    # :return: the candidate object with specific index
    def get_candidate_dict(self, index):
        return {C.L_NET:self.l_dim[index],
            C.L_ACTIVATION:self.a_functions[index],
            C.L_ETA:self.eta[index],
            C.L_TAU: self.tau[index],
            C.L_G_CLIPPING: self.g_clipping[index],
            C.L_REG:self.reg[index],
            C.L_DIMBATCH:self.dim_batch[index],
            C.L_MOMENTUM: self.momentum[index],
            C.L_EPOCHS:self.epochs[index],
            C.L_SHUFFLE:self.batch_shuffle[index],
            C.L_EPS: self.eps[index],
            C.L_DISTRIBUTION:self.distribution[index],
            C.L_BIAS:self.bias[index],
            C.L_SEED: self.seed[index],
            C.L_CLASSIFICATION:self.classification[index],
            C.L_EARLYSTOP:self.early_stop[index],
            C.L_PATIENCE: self.patience[index],
            C.L_TRESHOLD_VARIANCE:self.treshold_variance[index]
        }
    
    # :return: all the candidates objects list
    def get_all_candidates_dict(self):
        candidates:list = []
        for index in range (self.count):
            candidates.append(self.get_candidate_dict(index))
        return candidates
    
    # :param: default is the name of the config for initial demo
    # :param: theta is the configuration object
    def set_project_hyperparameters(self, default:str = C.MONK, theta = None):
        """
        :param default: it's the default config that was used
        :return: setting of parameters for the config theta that will use by model 
        """
        if theta != None:
            self.l_dim = theta[C.L_NET]
            self.a_functions = theta[C.L_ACTIVATION]
            self.eta = theta[C.L_ETA]
            self.momentum = theta[C.L_MOMENTUM]
            self.reg = theta[C.L_REG]
            self.dim_batch = theta[C.L_DIMBATCH]
            self.tau = theta[C.L_TAU]
            self.g_clipping = theta[C.L_G_CLIPPING]
            self.patience = theta[C.L_PATIENCE]
            self.batch_shuffle=theta[C.L_SHUFFLE]
            self.early_stop = theta[C.L_EARLYSTOP]
            self.eps = theta[C.L_EPS]
            self.distribution = theta[C.L_DISTRIBUTION]
            self.bias = theta[C.L_BIAS]
            self.epochs = theta[C.L_EPOCHS]
            self.classification= theta[C.L_CLASSIFICATION]
            self.treshold_variance = theta[C.L_TRESHOLD_VARIANCE]
            self.seed=theta[C.L_SEED]
        elif default == C.MONK:
            self.l_dim = [[17,2,1],[17,4,1]]
            self.a_functions = [[C.RELU,C.TANH],[C.TANH,C.TANH]]#,[C.RELU, C.SIGMOID]
            self.eta = [0.1,0.05]
            self.momentum = [(C.NESTEROV, 0.9),(C.CLASSIC, 0.7),]
            self.reg = [ (False, False),(C.LASSO, 0.02), (C.TIKHONOV, 0.02) ]#(TIKHONOV, 0.01), (LASSO, 0.01), (False, False)
            self.dim_batch = [0,15]
            self.tau = [(300,0.01)]
            self.patience = [200]
            self.early_stop = True
            self.eps = [0.1, 0.3]
            self.distribution = [C.UNIFORM,C.BASIC]
            self.bias = [0]
            self.epochs = [1000]
            self.classification = True
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
        elif default == C.CUP:
            self.l_dim = [[10,4,4,3]]#,[10,32,16,3],[10,64,32,16,3],[10,64,32,16,8,3]]
            self.a_functions = [[C.TANH,C.TANH,C.IDENTITY]]#,[C.RELU,C.RELU,C.IDENTITY],[C.RELU,C.RELU,C.RELU,C.IDENTITY],[C.TANH,C.RELU,C.RELU,C.IDENTITY]]
            self.eta=[0.2, 0.1]
            self.momentum=[(False,False) ]#, (C.NESTEROV,0.5) , (C.CLASSIC,0.5),(C.NESTEROV,0.9) , (C.CLASSIC,0.9) ]
            self.reg=[(False,False)]#, (C.TIKHONOV,0.01),  (C.LASSO,0.01) ]
            self.dim_batch=[0, 200]
            self.tau=[(1000,0.05)]
            self.g_clipping=[(True,0.5)]#, (1000,0.01)]
            self.patience=[200]
            self.eps=[0.2, 0.7]
            self.early_stop = True
            self.distribution=[C.UNIFORM]#,C.BASIC]
            self.bias = [0]
            self.classification = False
            self.epochs = [2000]
            self.treshold_variance = [1.e-6]
        else: 
            raise ValueError(f'Invalid {default} and {theta}')