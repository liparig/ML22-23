#!/usr/bin/env python
# coding: utf-8

from activationFunctions  import activations
from activationFunctions  import derivatives
import costants as C
from lossFunctions import loss
from regularization import regularization
from metrics import dnn_metrics
import math
import numpy as np
from numpy.random import Generator, PCG64

"""Didactic Neural Network"""
class DidacticNeuralNetwork:

    """
    Constructor
    :param l_dim: list of the layer network dimension
    :param a_functions: list of the activation function of each layer, if list len is 1 will be used 1 activaton function for all the network 
    :param l_function: name of loss to apply to the network
    :param eta: learning rate. Default: 0.5
    :param tau: tuple (tau,eta_tau) if set learning rate decay will be used default False. example (200, 0.005). Default: False
    :param epochs: training epochs
    :param batch_shuffle: if True and it's a mini-batch train each epoch batch are shuffled. Default: True
    :param reg: tuple with name of regularization type and lambda value default:empty esemple: reg=('tikhonov',0.01)
    :param momentum: tuple ("name",alpha) If set "classic" or "nesterov" will be applied the momentum to the delta. Default: ("",0) 
    :**kwargs: may contain arguments for the network initialization: bias values, seed for random numpy functions, eps for weight reduction.

    Step:
    1. init the hyperparameters of the NN network
    2. check dimensions of layers array and activation functions array
    3. init random weights 
    """
    def __init__(self, l_dim:list[int] = None, a_functions:list[str] = [C.SIGMOID], l_function:str = C.MSE, eta:float = C.ETA, 
                 tau = C.TAU, epochs:int = C.EPOCHS, batch_shuffle:bool = C.BATCH_SHUFFLE, reg = C.REG, momentum = C.MOMENTUM, 
                 classification:bool = False, early_stop:bool = True, patience:int = C.PATIENCE, treshold_variance:float = C.TRESHOLDVARIANCE, 
                 dim_batch:int = C.BATCH, plot = None, seed:int = C.R_SEEDS, **kwargs):
        self.gen = Generator(PCG64(seed))
        self.a_functions = a_functions
        self.__check_init__(l_dim, a_functions)
        self.l_dim = l_dim
        self.l_function = loss[l_function]
        self.wb = self.init_wb(l_dim, **kwargs)
        self.net = {}
        self.out = {}
        self.deltaOld = {}
        self.metrics = dnn_metrics()
        #save eta and momentum in self variable
        self.eta = eta
        self.momentum = momentum[0]
        self.alpha = momentum[1]
        #if reg set initialize regularization class and lambda value
        if reg[0]:
            self.regular = regularization[reg[0]]
            self.lambdar = reg[1]
        else:
            self.regular = False
        #dynamic decay learning hyperparameters
        if tau[0]:
            self.learning_decay = True
            self.decay_step = tau[0]
            self.eta_tau = tau[1]
        else:
            self.learning_decay = False
        
        self.epochs = epochs
        self.shuffle = batch_shuffle
        self.classification = classification
        self.early_stop = early_stop
        self.patience = patience
        self.treshold_variance = treshold_variance
        self.dim_batch = dim_batch
        self.plot = plot
        
    '''
    Check the network initialization params
    :param: later dimension
    :param: array activation funcions name
    :return: void
    '''
    def __check_init__(self, l_dim:list[int], a_functions):        
        num_layers:int = len(l_dim)
        num_functions:int = len(a_functions)
        if num_layers <= 2:
            raise Exception(f'l_dimension should be at list, with layer number, greater than 2 [input,hidden,output]. was: {num_layers}')
        if num_functions == 1:
           self.a_functions=[a_functions[0] for _ in range(1, num_layers)]
        elif num_functions != num_layers-1:
            raise Exception(f'Activation functions dimension is <> 1 or <> of layers number. Activation functions was: {num_functions} number of layer:{num_layers-1}') 

    '''
    Initialization of the network layer with Weights and Bias.
    l_dim value example for a network with 6 inputs 2 hidden layer with 3 and 5 units and 1 output unit l_dim will be: [6,3,5,1]
    :param l_dim: l_dim array contains the network layer dimension. Must includes also the input layer and the output layer. 
    :param seed: parametro seed per la funzione random così da avere numeri casuali uguali per test sugli stessi dati.
    :param eps: eps value  between 0 and 1 for  Weight decrement default 0.1 if random distribution, if uniform it's the high extreme of interval
    :param bias: - value for the biases
    '''
    def init_wb(self, l_dim, distribution:str = C.UNIFORM, eps:float = C.EPS, bias = C.BIAS):
        wb = {}
        #num_layers network deep
        num_layers:int = len(l_dim)
        for l in range(1, num_layers):
            name_layer:str = str(l)
            if distribution == C.UNIFORM:
              wb[f'W{name_layer}'] = self.gen.uniform(low = -eps, high = eps, size = (l_dim[l], l_dim[l-1]))
            elif distribution == C.BASIC:
                wb[f'W{name_layer}'] = self.gen.uniform(low = -1/l_dim[0], high = 1/l_dim[0], size = (l_dim[l], l_dim[l-1]))
            else:
                #Initialization of random weitghs ruled by eps
                wb[f'W{name_layer}'] = self.gen.random(size = (l_dim[l], l_dim[l-1])) * eps
            #Initialization of bias of the layer
            wb[f'b{name_layer}'] = np.full((l_dim[l], 1),bias)
            
        return wb
        
    def linear(self, w, X,b):
        return np.dot(w, X.T) + b
    '''
    Compute the forward propagation on the network. Update the network dimension and nets and outputs of the layers
    :parm inputs: Inputs values matrix to predict if None the last training input will be execute.
    :parm update: If true update the network nets and outputs of the inputs. Default:False 
    :return in_out: return the output value of the network 
    '''
    def forward_propagation(self, inputs, update:bool = False, nesterov:bool=False):            
        in_out = inputs
        #forward propagation of the input pattern in the network
        interim=""
        if nesterov:
            interim="_"
        for l in range(1, len(self.l_dim)):
            name_layer:str = str(l)
            w = self.wb[f'W{interim}{name_layer}'] # on the row there are the weights for 
            b = self.wb[f'b{interim}{name_layer}']
            #apply linear function
            net = np.asarray(self.linear(w, in_out, b))
            #apply activation function to the net
            af = activations[self.a_functions[l-1]]
            #traspose the result row unit column pattern
            in_out = np.asarray(af(net).T)
            #if update true record the net and out of the units on the layer
            if update:
                self.out[f"out{interim}0"] = inputs
                self.net[f'net{interim}{name_layer}'] = net
                self.out[f'out{interim}{name_layer}'] = in_out
        return in_out
    '''
    Compute delta_k gradient of outputlayer
    :param y:  target pattern
    :param out: predicted pattern
    :net: matrix of nets values in the layer 
    :param d_activation: function f prime of activation funcion
    :return: Delta_k matrix contain output units delta. with row is p-th patterns and column is k-th output unit
    '''    
    def compute_delta_k(self, y, out, net, d_activation):
        #delta of loss function of pattern 
        dp = self.l_function.d_loss(self, y, out) 
        #derivative of the layer activation function
        f_prime = d_activation(net)
        #compute delta with a puntual multiplication
        delta_k = dp.T * f_prime
        #return delta transpose for use rows patterns
        return delta_k.T
    '''
    Compute delta_j gradient of hidden layer
    :param delta_in: the delta matrix backpropagated by the upper layer.
    :param w_layer: weight matrix of the layer W_kj: rows units column inputs 
    :param net: matrix of nets values in the layer 
    :param d_activation: function f prime of activation funcion
    :return: Delta_j matrix contain layer hidden units delta. with row is p-th patterns and column is j-th hidden unit
    '''    
    def compute_delta_j(self, delta_in, w_layer, net, d_activation):
        # matrix multiplication for delta_t
        dt = delta_in @ w_layer
        # Transpose and puntual multiplication of apply the derivative
        fprime = d_activation(net) # net is a vector with all the nets for the unit layer
        fprime = fprime if isinstance(fprime, int) else fprime.T
        # print(f'shape delta_in: {delta_in.shape}, w_layer: {w_layer.shape}, dt {dt.shape}')
        # input('premi')
        dj = dt * fprime
        return dj
    '''
    Compute the weigth update
    :param delta: array contains the delta matrix computed in the backpropagation
    :param pattern: number of pattern predicted  
    :return: void
    '''
    def update_wb(self, delta, pattern:float):
        p_eta:float = (self.eta) 
        for l in range(len(self.l_dim) - 1, 0, -1):
            name_layer:str = str(l)
            #print(f"Name {name_layer}",self.out[f"out{l-1}"].shape)
            deltaW = (delta[l-1].T @self.out[f"out{l-1}"])  
            deltaB = (np.sum(delta[l-1].T,axis=1,keepdims=True))
            
            #save the old gradient for the Nesterov momentum if needed
            if self.momentum == C.NESTEROV:
                self.deltaOld[f'wold{name_layer}'] = deltaW
                self.deltaOld[f'bold{name_layer}'] = deltaB

            deltaW = p_eta * deltaW
            deltaB = p_eta * deltaB

           
            #if momentum classic is set add the momentum deltaW
            if self.momentum == C.CLASSIC and f'wold{name_layer}' in self.deltaOld:
                deltaW += self.alpha*self.deltaOld[f'wold{name_layer}']
                deltaB += self.alpha*self.deltaOld[f'bold{name_layer}']
                #save the old gradient for the momentum if needed
                self.deltaOld[f'wold{name_layer}'] = deltaW
                self.deltaOld[f'bold{name_layer}'] = deltaB
            

            
            #if regularization is set subtract the penalty term.
            if self.regular:
                deltaW -= self.regular.derivative(self, self.lambdar, self.wb[f'W{name_layer}'])
                deltaB -= self.regular.derivative(self, self.lambdar, self.wb[f'b{name_layer}'])

           
            self.wb[f'W{name_layer}'] = np.add(self.wb[f'W{name_layer}'], deltaW) 
            self.wb[f'b{name_layer}'] = np.add(self.wb[f'b{name_layer}'], deltaB)


    '''
    Compute the back propagation algorithm
    :param y: target set
    :return: array delta with the gradients matrix of each layer. Last element is the last elements gradiend matrix
    '''
    def back_propagation(self, y):
        delta_t = []
        num_layers:int = len(self.l_dim) - 1
        interim:str = ""
        #Reverse loop of the network layer
        if self.momentum == C.NESTEROV and "wold1" in self.deltaOld:
                interim = "_"
        for l in range(num_layers, 0, -1):
            dt = 0
            name_layer:str = str(l)
            # print(f'name_layer 228 ddn: {name_layer}')
            #recover the activation function of the layer
            af = self.a_functions[num_layers-1]
            #if not the output layer compute gradients delta_j
            if l != num_layers:
                dt = self.compute_delta_j(delta_t[-1], self.wb[f'W{interim}{l+1}'], self.net[f'net{interim}{name_layer}'], derivatives[af])
            else:
                dt = self.compute_delta_k(y, self.out[f'out{interim}{name_layer}'], self.net[f'net{interim}{name_layer}'], derivatives[af])
            #append the gradient matrix
            # print(f'dt.shape {dt.shape}, dt {dt}')
            # input('premi')
            delta_t.append(dt)
        return delta_t[::-1]

    """
    :return: history_terror, history_tloss, validation_error. History Error of the train and validation error
    """
    def train(self, validation:bool = False):
        #initialize error/loss variable
        history_terror, history_tloss, validation_error = [], [], []
        metric_tr, metric_val = [], []
        c_metric:dict[str, list] = {
            #training
            't_misclassified': [],
            't_classified':[],
            't_accuracy':[],
            't_precision':[],
            't_recall':[],
            't_specificity':[],
            't_balanced':[],
            #validation
            'v_misclassified': [],
            'v_classified':[],
            'v_accuracy':[],
            'v_precision':[],
            'v_recall':[],
            'v_specificity':[],
            'v_balanced':[]
        }
        selfcontrol:int = 0
        eta_0:float = self.eta
        
        #reference dataset in new variable
        x_dev = self.dataset[C.INPUT_TRAINING]
        y_dev = self.dataset[C.OUTPUT_TRAINING]
       
        #train start    
        for epoch in range(self.epochs):
            #initialize variable for partial error and loss
            batch_terror, batch_tloss = 0, 0

            #if learning decay used update of eta
            if self.learning_decay and self.decay_step >= epoch:
                alpha = epoch / self.decay_step
                self.eta = (1 - alpha) * eta_0 + (alpha) * self.eta_tau
            
            #region MINIBATCH
            #if mini-batch and shuffled true index of the set are shuffled
            if self.dim_batch != self.dataset[C.NUM_POINT_X] and self.shuffle:
                newindex = list(range(self.dataset[C.NUM_POINT_X]))
                self.gen.shuffle(newindex)
                x_dev = x_dev[newindex]
                y_dev = y_dev[newindex]    
            #if mini-batch loop the divided input set
            batch_number=math.ceil(self.dataset[C.NUM_POINT_X] / self.dim_batch)
            for b in range(batch_number):
                #initialize penalty term for loss calculation of each mini-batch
                p_term = 0
                #initialize index for dataset partition
                start = b * self.dim_batch
                end = start + self.dim_batch
                batch_x = np.asarray(x_dev[start: end])
                batch_y = np.asarray(y_dev[start: end])
                #propagation on network layer
                out = self.forward_propagation(batch_x.copy(), update=True)
                #print("\n\n\nOUT\n\n\n",out,"\n\n\n\n----\n\n\n")
                #if it's used nesterov or regularization, walk the network for compute penalty term and intermediate w
                if self.regular or self.momentum == C.NESTEROV:        
                    for l in range(1, len(self.l_dim)):
                        if self.regular:
                            """Note that often the bias w0 is omitted from the regularizer (because its inclusion causes the results to be not independent from target shift/scaling) or it may be included but with its own regularization coefficient (see Bishop book, Hastie et al. book)"""
                            #p_term += self.regular.penalty(self, self.lambdar, self.wb[f"W{l}"]) + self.regular.penalty(self, self.lambdar, self.wb[f"b{l}"])
                            p_term += self.regular.penalty(self, self.lambdar, self.wb[f"W{l}"]) 

                        if self.momentum == C.NESTEROV and f"wold{l}" in self.deltaOld :
                            self.wb[f"W_{l}"] = self.wb[f"W{l}"] + (self.alpha*self.deltaOld[f"wold{l}"])
                            self.wb[f"b_{l}"] = self.wb[f"b{l}"] + (self.alpha*self.deltaOld[f"bold{l}"])  
                    if self.momentum == C.NESTEROV  and f"wold{l}" in self.deltaOld:
                        self.forward_propagation(batch_x.copy(), update=True, nesterov=True)              
                #compute delta using back propagation on target batch
                delta = self.back_propagation(batch_y)
                # call update weights function
                self.update_wb(delta, batch_x.shape[0])
                # update bacth error
                terror = (self.l_function.loss(self, batch_y, out))
                batch_terror += terror 
                batch_tloss += terror + p_term
            
            #endregion
            out = self.forward_propagation(x_dev, update=False)
            if self.regular:
                for l in range(1, len(self.l_dim)):
                    """Note that often the bias w0 is omitted from the regularizer (because its inclusion causes the results to be not independent from target shift/scaling) or it may be included but with its own regularization coefficient (see Bishop book, Hastie et al. book)"""
                    #p_term += self.regular.penalty(self, self.lambdar, self.wb[f"W{l}"]) + self.regular.penalty(self, self.lambdar, self.wb[f"b{l}"])
                    p_term = self.regular.penalty(self, self.lambdar, self.wb[f"W{l}"]) 
             #append the error and the loss (mean if min-bacth or stochastic)
            terror = (self.l_function.loss(self, y_dev, out))/out.shape[0]
            history_terror.append(terror)
            history_tloss.append(terror + p_term)
            #history_terror.append(terror)
            #history_tloss.append(terror + p_term)    
            out_t = self.forward_propagation(inputs = self.dataset[C.INPUT_TRAINING], update=False)
            #if there'is validation compute validation metric for regression and classification
            if validation:
                out_v = self.forward_propagation(inputs = self.dataset[C.INPUT_VALIDATION], update=False)
                validation_error.append(self.l_function.loss(self, self.dataset[C.OUTPUT_VALIDATION],out_v)/out_v.shape[0])
                if self.classification:
                    mbc_v = self.metrics.metrics_binary_classification(self.dataset[C.OUTPUT_VALIDATION],out_v,treshold=0.5)
                    c_metric['v_accuracy'].append(mbc_v[C.ACCURACY])
                    c_metric['v_precision'].append(mbc_v[C.PRECISION])
                    c_metric['v_recall'].append(mbc_v[C.RECALL])
                    c_metric['v_specificity'].append(mbc_v[C.SPECIFICITY])
                    c_metric['v_balanced'].append(mbc_v[C.BALANCED])
                    c_metric['v_misclassified'].append(mbc_v[C.MISSCLASSIFIED])
                    c_metric['v_classified'].append(mbc_v[C.CLASSIFIED])                
                    metric_val.append(mbc_v[C.ACCURACY])
                else:
                    metric_val.append(self.metrics.mean_euclidean_error(self.dataset[C.OUTPUT_VALIDATION], out_v))
                    
            if self.classification:
                mbc = self.metrics.metrics_binary_classification(self.dataset[C.OUTPUT_TRAINING],out_t,treshold=0.5)
                c_metric['t_accuracy'].append(mbc[C.ACCURACY])
                c_metric['t_precision'].append(mbc[C.PRECISION])
                c_metric['t_recall'].append(mbc[C.RECALL])
                c_metric['t_specificity'].append(mbc[C.SPECIFICITY])
                c_metric['t_balanced'].append(mbc[C.BALANCED])
                c_metric['t_misclassified'].append(mbc[C.MISSCLASSIFIED])
                c_metric['t_classified'].append(mbc[C.CLASSIFIED])
                metric_tr.append(mbc[C.ACCURACY])
            else:
                metric_tr.append(self.metrics.mean_euclidean_error(self.dataset[C.OUTPUT_TRAINING], out_t))
                        
            if epoch>1 and self.early_stop:
                if validation_error[-1] >= validation_error[-2] or np.square(validation_error[-2]-validation_error[-1]) < self.treshold_variance:
                    selfcontrol += 1
                    if self.patience == selfcontrol:
                        break
                else:
                    selfcontrol = 0
                    
        return {'error':history_terror,'loss':history_tloss, 'metric_tr':metric_tr, 'metric_val':metric_val, 'validation':validation_error, 'c_metrics':c_metric, 'epochs':epoch + 1}  

    '''
    Check problem in dataset inputs and add an error message to the exception:
    '''      
    def check_dimension(self, x, y, dim_batch, msg = "Error check_dimension line 360"):
        #Check output and input dimension 
        output_dim = y.ndim if y.ndim == 1 else y.shape[1] 
        input_dim = x.ndim if x.ndim == 1 else x.shape[1] 
        error = msg+" "
        if output_dim != self.l_dim[-1]:
            error+=f"Output dimension:{output_dim} <> Output Units:{self.l_dim[-1]}. "
        if input_dim != self.l_dim[0]:
            error+=f"Input dimension:{input_dim} <> Input units:{self.l_dim[0]}. "
        if x.shape[0] != y.shape[0]:
            error+=f"Number of pattern input:{x.shape[0]} <> target:{y.shape[0]}. "
        if dim_batch > 1 and dim_batch > x.shape[0]:
            error+=f"Number of batch:{dim_batch} > number of pattern:{x.shape[0]}. "
        if len(error) > len(msg+" "):
            raise Exception(error)
        return True

    """
    Execute the training of the network
    :param x_train: input training set
    :param y_train: targets for each input training pattern
    :param val_x: input validation set
    :param val_y: targets for each input validation pattern
    :param dim_batch: the dimension of mini-batch, if 0 is equal to batch, if 1 is stochastic/online update version
    :param **kwargs: extra params for the training
    """
    def fit(self, x_train, y_train, x_val = [], y_val = [], dim_batch = None):
        #initialize a dictionary contain dataset information
        self.dataset = {}
        self.dataset[C.NUM_POINT_X] = x_train.shape[0]
        self.dataset[C.INPUT_TRAINING]= x_train.copy()
        self.dataset[C.OUTPUT_TRAINING]= y_train.copy()
        
        validation:bool = False
        
        if  x_val.size != 0 and y_val.size != 0:
            self.dataset[C.INPUT_VALIDATION]= x_val.copy()
            self.dataset[C.OUTPUT_VALIDATION]= y_val.copy()
            validation = True
            
        
        if (dim_batch == None):
            dim_batch=self.dim_batch
        #check parameter batch exist and if minibacth, batch or stochastic
        
        if (dim_batch == 0):
            self.dim_batch = self.dataset[C.NUM_POINT_X]
        else:
            self.dim_batch = dim_batch

        self.check_dimension(x_train, y_train, dim_batch, msg = "[check_dimension] Error in Training datasets:")
        self.check_dimension(x_val, y_val, 1, msg = "[check_dimension] Error in Validation datasets:")

        return self.train(validation)

    