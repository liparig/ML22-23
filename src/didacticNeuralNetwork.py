#!/usr/bin/env python
# coding: utf-8

from activationFunctions  import activations
from activationFunctions  import derivatives
import costants as C
from lossFunctions import loss
from regularization import regularization
from metrics import DnnMetrics
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
        self.metrics = DnnMetrics()
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
        return np.add(np.dot(w, X.T) , b)
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
            net = self.linear(w, in_out, b)
            #apply activation function to the net
            af = activations[self.a_functions[l-1]]
            #traspose the result row unit column pattern
            in_out = af(net).T
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
        dj = dt * fprime
        return dj
    '''
    Compute the weigth update
    :param delta: array contains the delta matrix computed in the backpropagation
    :param pattern: number of pattern predicted  
    :return: void
    '''
    def update_wb(self, delta, gradients,pattern):
        for l in range(len(self.l_dim) - 1, 0, -1):
            name_layer:str = str(l)
            gradw = np.divide(gradients[l-1] , pattern)
            gradb = np.divide(np.sum(delta[l-1].T, axis=1, keepdims=True) ,pattern)

            
            #save the old gradient for the Nesterov momentum if needed
            if self.momentum == C.NESTEROV:
                self.deltaOld[f'wold{name_layer}'] = gradw
                self.deltaOld[f'bold{name_layer}'] = gradb

            deltaW = self.eta * gradw
            deltaB = self.eta * gradb

           
            #if momentum classic is set add the momentum deltaW
            if self.momentum == C.CLASSIC and f'wold{name_layer}' in self.deltaOld:
                deltaW = np.add(deltaW,self.alpha*self.deltaOld[f'wold{name_layer}'])
                deltaB = np.add(deltaB,self.alpha*self.deltaOld[f'bold{name_layer}'])
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
        gradients=[]
        num_layers:int = len(self.l_dim) - 1
        interim:str = ""
        if self.momentum == C.NESTEROV and "wold1" in self.deltaOld:
                interim = "_"
        #Reverse loop of the network layer
        for l in range(num_layers, 0, -1):
            dt = 0
            name_layer:str = str(l)
            #recover the activation function of the layer
            af = self.a_functions[l-1]
            #if not the output layer compute gradients delta_j
            if l != num_layers:
                dt = self.compute_delta_j(delta_t[-1], self.wb[f'W{interim}{l+1}'], self.net[f'net{interim}{name_layer}'], derivatives[af])
            else:
                dt = self.compute_delta_k(y, self.out[f'out{interim}{name_layer}'], self.net[f'net{interim}{name_layer}'], derivatives[af])
            #append the gradient matrix
            # print(f'dt.shape {dt.shape}, dt {dt}')
            # input('premi')
            #The problem of overflow is  here TODO!!!!
            gradW = dt.T @ self.out[f"out{interim}{l-1}"]
            delta_t.append(dt)
            gradients.append(gradW)

        return delta_t[::-1], gradients[::-1]

    """
    :return: history_terror, history_tloss, validation_error. History Error of the train and validation error
    """
    def train(self, validation:bool = False,test:bool=False):
        #initialize error/loss variable
        history_terror, history_tloss, validation_error,test_error = [], [], [],[]
        metric_tr, metric_val, metric_test = [], [],[]
        c_metric:dict[str, list] = {
            #training
            f'{C.TRAINING}_misclassified': [],
            f'{C.TRAINING}_classified':[],
            f'{C.TRAINING}_accuracy':[],
            f'{C.TRAINING}_precision':[],
            f'{C.TRAINING}_recall':[],
            f'{C.TRAINING}_specificity':[],
            f'{C.TRAINING}_balanced':[],
            #validation
            f'{C.VALIDATION}_misclassified': [],
            f'{C.VALIDATION}_classified':[],
            f'{C.VALIDATION}_accuracy':[],
            f'{C.VALIDATION}_precision':[],
            f'{C.VALIDATION}_recall':[],
            f'{C.VALIDATION}_specificity':[],
            f'{C.VALIDATION}_balanced':[],
            #test
            f'{C.TEST}_misclassified': [],
            f'{C.TEST}_classified':[],
            f'{C.TEST}_accuracy':[],
            f'{C.TEST}_precision':[],
            f'{C.TEST}_recall':[],
            f'{C.TEST}_specificity':[],
            f'{C.TEST}_balanced':[]
        }

        """#compute errors at epoch 0
        out_t = self.forward_propagation(inputs = self.dataset[C.INPUT_TRAINING], update=False)
        terror = (self.l_function.loss(self, self.dataset[C.OUTPUT_TRAINING], out_t)/out_t.shape[0])
        history_terror.append(terror)
        if self.regular:
            history_tloss.append(terror)
           
        self.training_metrics(metric_tr, c_metric, out_t, False)
        if validation:
            self.validation_metrics(validation_error, metric_val, c_metric)
        if test:
            self.test_metrics(test_error, metric_test, c_metric)
        """

        selfcontrol:int = 0
        eta_0:float = self.eta
        
        #reference dataset in new variable
        x_dev = self.dataset[C.INPUT_TRAINING]
        y_dev = self.dataset[C.OUTPUT_TRAINING]
        #if mini-batch loop the divided input set
        batch_number = math.ceil(self.dataset[C.NUM_PATTERN_X] / self.dim_batch)
        #train start    
        for epoch in range(self.epochs):
            #initialize variable for training partial error and loss
            batch_terror, batch_tloss = 0, 0

            #if learning decay used update of eta
            if self.learning_decay and self.decay_step >= epoch:
                self.eta_decay(eta_0, epoch)
            
            #region MINIBATCH
            #if mini-batch and shuffled true index of the set are shuffled
            if self.dim_batch != self.dataset[C.NUM_PATTERN_X] and self.shuffle:
                x_dev, y_dev = self.shuffle_dataset(x_dev, y_dev)    
            
            #batch or minibatch training
            for b in range(batch_number):
                #initialize penalty term for loss calculation of each mini-batch
                p_term:float = 0
                #initialize index for dataset partition
                batch_x, batch_y = self.extract_batch(x_dev, y_dev, b)
                #propagation on network layer
                self.forward_propagation(batch_x.copy(), update = True)
                #if it's used nesterov or regularization, walk the network for compute penalty term and intermediate w
                if self.regular or self.momentum == C.NESTEROV:        
                    for l in range(1, len(self.l_dim)):
                        if self.regular:
                           #Note that often the bias w0 is omitted from the regularizer (because its inclusion causes the results to be not independent from target shift/scaling) or it may be included but with its own regularization coefficient (see Bishop book, Hastie et al. book)
                            #p_term += self.regular.penalty(self, self.lambdar, self.wb[f"W{l}"]) + self.regular.penalty(self, self.lambdar, self.wb[f"b{l}"])
                            p_term += self.regular.penalty(self, self.lambdar, self.wb[f"W{l}"])
                        #Apply Nesterov momentum computing the interim W_=w+ alpha*oldw
                        if self.momentum == C.NESTEROV and f"wold{l}" in self.deltaOld :
                            self.wb[f"W_{l}"] = self.wb[f"W{l}"] + (self.alpha*self.deltaOld[f"wold{l}"])
                            self.wb[f"b_{l}"] = self.wb[f"b{l}"] + (self.alpha*self.deltaOld[f"bold{l}"])  
                    #compute the output of the interim w for delta w computation
                    if self.momentum == C.NESTEROV  and f"wold{l}" in self.deltaOld:
                        self.forward_propagation(batch_x.copy(), update=True, nesterov=True)              
                #compute delta using back propagation on target batch
                delta,gradients= self.back_propagation(batch_y)
                # call update weights function
                self.update_wb(delta,gradients,batch_x.shape[0])
                # update bacth error
                out_t = self.forward_propagation(batch_x, update=False)

                terror = self.l_function.loss(self, batch_y, out_t)
                terror = np.sum(terror)/out_t.shape[0]
                batch_terror += terror 
                if self.regular:
                    batch_tloss += terror + p_term
            
            #endregion 
                    
           
            #append the error and the loss (mean if min-bacth or stochastic)
            history_terror.append(batch_terror/(b+1))
            if self.regular:
                history_tloss.append((batch_tloss)/(b+1))
           
            self.training_metrics(metric_tr, c_metric, out_t, batch_number>1)

            #if there'is validation or test compute validation and test metric for regression and classification
            if validation:
                self.validation_metrics(validation_error, metric_val, c_metric)
            if test:
                self.test_metrics(test_error, metric_test, c_metric)                       

            
            if epoch>1 and self.early_stop:
                if validation_error[-1] >= validation_error[-2] or np.square(validation_error[-1]-validation_error[-2]) < self.treshold_variance:
                    selfcontrol += 1
                    if self.patience == selfcontrol:
                        break
                else:
                    selfcontrol = 0
        result={}
        result['error']=history_terror
        result['loss']=history_tloss
        result['metric_tr']=metric_tr
        result['c_metrics']=c_metric
        result['epochs']=epoch + 1
        
        if validation:
            result['validation']=validation_error
            result['metric_val']=metric_val
        if test:
            result['test']=test_error
            result['metric_test']=metric_test
       
        return result

    def training_metrics(self, metric_tr, c_metric, out_t, evaluate_out_t):
        if evaluate_out_t:
            out_t = self.forward_propagation(inputs = self.dataset[C.INPUT_TRAINING], update=False)
        if self.classification:
            self.append_binary_classification_metric(c_metric, out_t,self.dataset[C.OUTPUT_TRAINING],treshold=0.5,dataset=C.TRAINING)
            self.metrics.metrics_binary_classification(self.dataset[C.OUTPUT_TRAINING],out_t,treshold=0.5)
            metric_tr.append(c_metric[f'{C.TRAINING}_accuracy'][-1])
        else:
            metric_tr.append(self.metrics.mean_euclidean_error(self.dataset[C.OUTPUT_TRAINING], out_t))

    def validation_metrics(self, validation_error, metric_val, c_metric):
        out_v = self.forward_propagation(inputs = self.dataset[C.INPUT_VALIDATION], update=False)
        v_error=np.sum(self.l_function.loss(self, self.dataset[C.OUTPUT_VALIDATION],out_v))
        v_error=v_error/out_v.shape[0]
        validation_error.append(v_error)
        if self.classification:
            self.append_binary_classification_metric(c_metric, out_v,self.dataset[C.OUTPUT_VALIDATION],treshold=0.5,dataset=C.VALIDATION)
            metric_val.append(c_metric[f'{C.VALIDATION}_accuracy'][-1])
        else:
            metric_val.append(self.metrics.mean_euclidean_error(self.dataset[C.OUTPUT_VALIDATION], out_v))
            
    def test_metrics(self, test_error, metric_test, c_metric):
        out_test = self.forward_propagation(inputs = self.dataset[C.INPUT_TEST], update=False)
        test_loss=np.sum(self.l_function.loss(self, self.dataset[C.OUTPUT_TEST],out_test))
        test_loss=test_loss/out_test.shape[0]
        test_error.append(test_loss)
        if self.classification:
            self.append_binary_classification_metric(c_metric, out_test,self.dataset[C.OUTPUT_TEST],treshold=0.5,dataset=C.TEST)
            metric_test.append(c_metric[f'{C.TEST}_accuracy'][-1])
        else:
            metric_test.append(self.metrics.mean_euclidean_error(self.dataset[C.OUTPUT_TEST], out_test))
    
    #dataset can be C.VALIDATION or 'C.TRANING'
    def append_binary_classification_metric(self, c_metric, predicted, target, treshold=0.5, dataset = C.VALIDATION):
        mbc = self.metrics.metrics_binary_classification(target,predicted,treshold)
        c_metric[f'{dataset}_accuracy'].append(mbc[C.ACCURACY])
        c_metric[f'{dataset}_precision'].append(mbc[C.PRECISION])
        c_metric[f'{dataset}_recall'].append(mbc[C.RECALL])
        c_metric[f'{dataset}_specificity'].append(mbc[C.SPECIFICITY])
        c_metric[f'{dataset}_balanced'].append(mbc[C.BALANCED])
        c_metric[f'{dataset}_misclassified'].append(mbc[C.MISSCLASSIFIED])
        c_metric[f'{dataset}_classified'].append(mbc[C.CLASSIFIED])                

    def extract_batch(self, x_dev, y_dev, b):
        start = b * self.dim_batch
        end = start + self.dim_batch
        batch_x = np.asarray(x_dev[start: end])
        batch_y = np.asarray(y_dev[start: end])
        return batch_x,batch_y

    def shuffle_dataset(self, x_dev, y_dev):
        newindex = list(range(self.dataset[C.NUM_PATTERN_X]))
        self.gen.shuffle(newindex)
        x_dev = x_dev[newindex]
        y_dev = y_dev[newindex]
        return x_dev, y_dev

    def eta_decay(self, eta_0:float, epoch:int):
        alpha = epoch / self.decay_step
        self.eta = (1 - alpha) * eta_0 + (alpha) * self.eta_tau 

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
    def fit(self, x_train, y_train, x_val = [], y_val = [],x_test = [], y_test = [], dim_batch = None):
        #initialize a dictionary contain dataset information
        self.dataset = {}
        self.dataset[C.NUM_PATTERN_X] = x_train.shape[0]
        self.dataset[C.INPUT_TRAINING]= x_train.copy()
        self.dataset[C.OUTPUT_TRAINING]= y_train.copy()
        
        validation:bool = False
        
        if  len(x_val) > 0 and len(y_val)> 0:
            self.dataset[C.INPUT_VALIDATION]= x_val.copy()
            self.dataset[C.OUTPUT_VALIDATION]= y_val.copy()
            self.check_dimension(x_val, y_val, 1, msg = "[check_dimension] Error in Validation datasets:")
            validation = True
            
        test:bool = False
        
        if  len(x_test)> 0 and len(y_test)> 0:
            self.dataset[C.INPUT_TEST]= x_test.copy()
            self.dataset[C.OUTPUT_TEST]= y_test.copy()
            self.check_dimension(x_test, y_test, 1, msg = "[check_dimension] Error in Test datasets:")
            test = True
            
        
        if (dim_batch == None):
            dim_batch=self.dim_batch
        #check parameter batch exist and if minibacth, batch or stochastic
        
        if (dim_batch == 0):
            self.dim_batch = self.dataset[C.NUM_PATTERN_X]
        else:
            self.dim_batch = dim_batch

        self.check_dimension(x_train, y_train, dim_batch, msg = "[check_dimension] Error in Training datasets:")
        
        return self.train(validation,test)

    