#!/usr/bin/env python
# coding: utf-8

from activationFunctions  import activations
from activationFunctions  import derivatives
from lossFunctions import loss
from regularization import regularization
from metrics import DNN_metrics
import dnn_plot
import math
import numpy as np

#DEFINE COSTANT OF DEFAULTS VALUES
R_SEEDS=30
EPS=0.7
BIAS=0
SIGMOID=["sigmoid"]
MSE="mse"
BATCH=0
ETA=0.5
TAU=(False,False)
REG=(False,False)
EPOCHS=200
MOMENTUM=("",0)
PATIENCE=20
EARLYSTOP=True
TRESHOLDVARIANCE=1.e-6

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
    """
    def __init__(self, l_dim, a_functions=SIGMOID, l_function=MSE, eta=ETA,tau=TAU, epochs=EPOCHS, batch_shuffle=True, reg=REG,momentum=MOMENTUM,classification=False, early_stop=True, patience=PATIENCE,treshold_variance=TRESHOLDVARIANCE ,dim_batch=0,plot=None,**kwargs):
        self.a_functions=a_functions
        self.__checkInit__(l_dim, a_functions)
        self.l_dim=l_dim
        self.l_function=loss[l_function]
        self.wb=self.init_wb(l_dim,**kwargs)
        self.net={}
        self.out={}
        self.deltaOld={}
        self.metrics=DNN_metrics()
        #save eta and momentum in self variable
        self.eta=eta
        self.momentum=momentum[0]
        self.alpha=momentum[1]
         #if reg set initialize regularization class and lambda value
        if reg[0]:
            self.regular=regularization[reg[0]]
            self.lambdar=reg[1]
        else:
            self.regular=False
        self.epochs=epochs
        if tau[0]:
            self.learning_decay=True
            self.decay_step=tau[0]
            self.eta_tau=tau[1]
        else:
            self.learning_decay=False
        self.shuffle=batch_shuffle
        self.classification=classification
        self.early_stop=early_stop
        self.patience=patience
        self.treshold_variance=treshold_variance
        self.dim_batch=dim_batch
        self.plot=plot

        
    '''
    Check the network initialization params
    :param: later dimension
    :param: array activation funcions name
    :return: void
    '''
    def __checkInit__(self,l_dim, a_functions):        
        if len(l_dim)<=2:
            raise Exception(f'l_dimension should be at list with layer number greater than 2 [input,hidden,output]. was: {len(l_dim)}')
        if len(a_functions)==1:
           self.a_functions=[a_functions[0] for _ in range(1,len(l_dim))]
        elif len(a_functions)!=len(l_dim)-1:
            raise Exception(f'Activation functions dimension is <> 1  or <> of layers number. Activation functions was: {len(a_functions)} number of layer:{len(l_dim)-1}')
        return 

    '''
    Initialization of the network layer with Weights and Bias.
    l_dim value example for a network with 6 inputs 2 hidden layer with 3 and 5 units and 1 output unit l_dim will be: [6,3,5,1]
    :param l_dim: l_dim array contains the network layer dimension. Must includes also the input layer and the output layer. 
    :param seed: parametro seed per la funzione random così da avere numeri casuali uguali per test sugli stessi dati.
    :param eps: eps value  between 0 and 1 for  Weight decrement default 0.1 if random distribution, if uniform it's the high extreme of interval
    :param bias: - value for the biases
    '''
    def init_wb(self,l_dim, distribution='uniform',seed=R_SEEDS,eps=EPS,bias=BIAS,):
        np.random.seed(seed)
        wb = {}
        #L network deep
        L = len(l_dim)
        for l in range(1, L):
            if distribution=='uniform':
               wb['W'+str(l)] = np.random.uniform(low=-eps, high=eps,size=(l_dim[l],l_dim[l-1]))
            else:
                #Initialization of random weitghs ruled by eps
                wb['W'+str(l)] = np.random.randn(l_dim[l],l_dim[l-1]) * eps
            #Initialization of bias of the layer
            wb['b'+str(l)] = np.full((l_dim[l], 1),bias)
        return wb
        
    def linear(self,w,X,b):
        return np.dot(w,X.T)+b
    '''
    Compute the forward propagation on the network. Update the network dimension and nets and outputs of the layers
    :parm inputs: Inputs values matrix to predict if None the last training input will be execute.
    :parm update: If true update the network nets and outputs of the inputs. Default:False 
    :return in_out: return the output value of the network 
    '''
    def forward_propagation(self, inputs, update=False):            
        in_out=inputs
        #forward propagation of the inputta pattern in the network
        for l in range(1,len(self.l_dim)):
            w=self.wb["W"+str(l)]
            b=self.wb["b"+str(l)]
            #apply linear function
            net = np.asarray( self.linear(w,in_out,b))
            #apply activation function to the net
            af=activations[self.a_functions[l-1]]
            #traspose the result row unit column pattern
            in_out = np.asarray(af(net).T)
            #if update true record the net and out of the units on the layer
            if update:
                self.out["out0"]=inputs
                self.net["net"+str(l)]=net
                self.out["out"+str(l)]=in_out
        return in_out
    '''
    Compute delta_k gradient of outputlayer
    :param y:  target pattern
    :param out: predicted pattern
    :net: matrix of nets values in the layer 
    :param d_activation: function f prime of activation funcion
    :return: Delta_k matrix contain output units delta. with row is p-th patterns and column is k-th output unit
    '''    
    def compute_delta_k(self,y,out,net,d_activation):
        #delta of loss function of pattern 
        dp=self.l_function.d_loss(y,out)
        #derivative of the layer activation function
        f_prime=d_activation(net)
        #compute delta with a puntual multiplication
        delta_k=dp.T*f_prime
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
    def compute_delta_j(self,delta_in,w_layer,net,d_activation):
        # matrix multiplication for delta_t
        dt=delta_in@w_layer
        # Transpose and puntual multiplication of apply the derivative
        fprime=d_activation(net)
        fprime= fprime if type(fprime)==int else fprime.T

        dj=dt * fprime
        return dj
    '''
    Compute the weigth update
    :param delta: array contains the delta matrix computed in the backpropagation
    :param pattern: number of pattern predicted  
    :return: void
    '''
    def update_wb(self,delta,pattern):
        p_eta=(self.eta/pattern)
        for l in range(len(self.l_dim)-1,0,-1):
            deltaW=-p_eta*((delta[l-1].T@self.out["out"+str(l-1)]))
            deltaB=-p_eta*(np.sum(delta[l-1]))
            #if regularization is set subtract the penalty term
            if self.regular:
                deltaW-= self.regular.derivative(self.lambdar,self.wb['W'+str(l)])
                deltaB-= self.regular.derivative(self.lambdar,self.wb['b'+str(l)])
            #if momentum classic is set add the momentum deltaW
            if self.momentum=='classic' and 'wold'+str(l) in self.deltaOld:
                deltaW+= self.alpha*self.deltaOld['wold'+str(l)]
                deltaB+= self.alpha*self.deltaOld['bold'+str(l)]

            self.wb['W'+str(l)]= self.wb['W'+str(l)] + deltaW 
            self.wb['b'+str(l)]= self.wb['b'+str(l)] + deltaB
            #save the old gradient for the momentum if needed
            self.deltaOld['wold'+str(l)]=deltaW
            self.deltaOld['bold'+str(l)]=deltaB
        return 

    '''
    Compute the back propagation algorithm
    :param y: target set
    :return: array delta with the gradients matrix of each layer. Last element is the last elements gradiend matrix
    '''
    def back_propagation(self,y):
        delta_t=[]
        L=len(self.l_dim)-1
        #Reverse loop of the network layer
        for l in range(L,0,-1):
            dt=0
            #recover the activation function of the layer
            af=self.a_functions[L-1]
            #if not the output layer compute gradients delta_j
            if l!=L:
                dt = self.compute_delta_j(delta_t[-1],self.wb['W'+str(l+1)],self.net["net"+str(l)],derivatives[af])
            else:
                dt = self.compute_delta_k(y,self.out["out"+str(l)],self.net["net"+str(l)],derivatives[af])
            #append the gradient matrix
            delta_t.append(dt)
        return delta_t[::-1]

    """
    :return: history_terror, history_tloss, validation_error. History Error of the train and validation error
    """
    def train(self ):
        #initialize error/loss variable
        history_terror, history_tloss, validation_error = [], [],[]
        metric_tr, metric_val=[],[]
        c_metric={
            't_misclassified': [],
            't_classified':[],
            't_accuracy':[],
            't_precision':[],
            't_recall':[],
            't_specificity':[],
            't_balanced':[],
            'v_misclassified': [],
            'v_classified':[],
            'v_accuracy':[],
            'v_precision':[],
            'v_recall':[],
            'v_specificity':[],
            'v_balanced':[]
        }
        selfcontrol=0
        eta_0=self.eta
        
        #reference dataset  in new variable
        x_dev = self.dataset['x_train']
        y_dev = self.dataset['y_train']

       
        #train start    
        for e in range(self.epochs):
            #initialize variable for partial error and loss
            batch_terror ,batch_tloss=0,0
            #if learning decay used update of eta
            if self.learning_decay and self.decay_step >=e:
                self.eta=(1-(e/self.decay_step))*eta_0 + (e/self.decay_step)* self.eta_tau
            #if mini-batch and shuffled true  index of the set are shuffled
            if self.dim_batch != self.dataset['p_tx'] and self.shuffle:
                newindex = list(range(self.dataset['p_tx']))
                np.random.shuffle(newindex)
                x_dev = x_dev[newindex]
                y_dev = y_dev[newindex]    
            #if mini-batch  loop the divided input set
            for b in range(math.ceil(self.dataset['p_tx'] / self.dim_batch)):
                #initialize penalty term for loss calculation of each mini-batch
                p_term=0
                #initialize index for dataset partition
                start = b * self.dim_batch
                end = start + self.dim_batch
                batch_x = np.asarray(x_dev[start: end])
                batch_y = np.asarray(y_dev[start: end])
                #propagation on network layer
                out=self.forward_propagation(batch_x.copy(),update=True)
                #print("\n\n\nOUT\n\n\n",out,"\n\n\n\n----\n\n\n")
                #if it's used nesterov or regularization, walk the network for compute penalty term and intermediate w
                if self.regular or self.momentum=='nesterov':        
                    for l in range(1,len(self.l_dim)):
                        if self.regular:
                            p_term+=self.regular.penalty(self.lambdar,self.wb["W"+str(l)])+self.regular.penalty(self.lambdar,self.wb["b"+str(l)])
                        if self.momentum=='nesterov' and "wold"+str(l) in self.deltaOld :
                            self.wb["W"+str(l)]=self.wb["W"+str(l)]+ (self.alpha*self.deltaOld["wold"+str(l)])
                            self.wb["b"+str(l)]=self.wb["b"+str(l)]+ (self.alpha*self.deltaOld["bold"+str(l)])                
                #compute delta using back propagation on target batch
                delta=self.back_propagation(batch_y)
                # call update weights function
                self.update_wb(delta,batch_x.shape[0])
                # update bacth error
                terror=(self.l_function.loss(batch_y, out))
                batch_terror+=terror
                batch_tloss+=terror+p_term
            #append the error and the loss (mean if min-bacth or stochastic)
            history_terror.append(batch_terror/(b+1))
            history_tloss.append(batch_tloss/(b+1))
            out_t=self.forward_propagation(inputs=self.dataset['x_train'],update=False)
            out_v=self.forward_propagation(inputs=self.dataset['x_val'],update=False)
            validation_error.append(self.l_function.loss(self.dataset['y_val'],out_v))
            if self.classification:
                mbc=self.metrics.metrics_binary_classification(self.dataset['y_train'],out_t,treshold=0.5)
                mbc_v=self.metrics.metrics_binary_classification(self.dataset['y_val'],out_v,treshold=0.5)
                c_metric['t_accuracy'].append(mbc['accuracy'])
                c_metric['t_precision'].append(mbc['precision'])
                c_metric['t_recall'].append(mbc['recall'])
                c_metric['t_specificity'].append(mbc['specificity'])
                c_metric['t_balanced'].append(mbc['balanced'])
                c_metric['t_misclassified'].append(mbc['misclassified'])
                c_metric['v_classified'].append(mbc['classified'])
                c_metric['v_accuracy'].append(mbc_v['accuracy'])
                c_metric['v_precision'].append(mbc_v['precision'])
                c_metric['v_recall'].append(mbc_v['recall'])
                c_metric['v_specificity'].append(mbc_v['specificity'])
                c_metric['v_balanced'].append(mbc_v['balanced'])
                c_metric['v_misclassified'].append(mbc_v['misclassified'])
                c_metric['v_classified'].append(mbc_v['classified'])
                metric_tr.append(mbc['accuracy'])
                metric_val.append(mbc_v['accuracy'])
            else:
                metric_tr.append(self.metrics.mean_euclidean_error(self.dataset['y_train'],out_t))
                metric_val.append(self.metrics.mean_euclidean_error(self.dataset['y_val'],out_v))
            if e>=19 and self.early_stop:
                if np.var(history_terror[-20:])<self.treshold_variance:
                    selfcontrol+=1
                    if self.patience==selfcontrol:
                        break
                else:
                    selfcontrol=0
        if self.plot!=None:
                path=self.plot if type(self.plot)==str else None
                if self.classification:
                    ylim=(0., 1.)
                else:
                    ylim=(0., 5.)
                dnn_plot.plot_curves(history_terror, validation_error,metric_tr,metric_val,lbl_tr="Training",lbl_vs="Validation",path=path,ylim=ylim)
        return {'error':history_terror,'loss':history_tloss,'mee':metric_tr,'mee_v':metric_val,'validation': validation_error,'c_metrics':c_metric, 'epochs':e+1}  

    '''
    Check problem in dataset inputs and add an error message to the exception:
    '''      
    def check_dimension(self, x,y,dim_batch,msg=""):
        #Check output and input dimension 
        output_dim = y.ndim if y.ndim == 1 else y.shape[1] 
        input_dim = x.ndim if x.ndim == 1 else x.shape[1] 
        error=msg+" "
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
    :param dim_batch: the dimension of mini-batch, if 0 is equal to batch, if 1  is stochastic
    :param **kwargs:  param for the training
    """
    def fit(self, x_train, y_train, x_val, y_val, dim_batch=BATCH):
        self.check_dimension(x_train,y_train,dim_batch,msg="Error in Training datasets:")
        self.check_dimension(x_val,y_val,1,msg="Error in Validation datasets:")

        #initialize a dictionary contain dataset information
        self.dataset={}
        self.dataset['p_tx'] = x_train.shape[0]
        self.dataset['x_train']= x_train.copy()
        self.dataset['y_train']= y_train.copy()
        self.dataset['x_val']= x_val.copy()
        self.dataset['y_val']= y_val.copy()

        #check if minibacth, batch or stochastic
        if dim_batch == 0:
            dim_batch = self.dataset['p_tx']
        self.dim_batch=dim_batch
        
        return self.train()

    