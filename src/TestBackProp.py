import numpy as np
import costants as C
from numpy.random import Generator, PCG64
from activationFunctions  import activations
from activationFunctions  import derivatives
from lossFunctions import MSE

def init_wb(l_dim, distribution:str = C.UNIFORM, eps:float = C.EPS, bias = C.BIAS):
        wb = {}
        gen = Generator(PCG64(123))
        #num_layers network deep
        num_layers:int = len(l_dim)
        for l in range(1, num_layers):
            name_layer:str = str(l)
            if distribution == C.UNIFORM:
               wb[f'W{name_layer}'] = gen.uniform(low = -eps, high = eps, size = (l_dim[l], l_dim[l-1]))
            else:
                #Initialization of random weitghs ruled by eps
                wb[f'W{name_layer}'] = gen.random(size = (l_dim[l], l_dim[l-1])) * eps
            #Initialization of bias of the layer
            wb[f'b{name_layer}'] = np.full((l_dim[l], 1),bias)
        return wb

def linear(w, X,b):
        return np.dot(w, X.T) + b

def init_global():
    global out
    global net 
    global deltaOld
    out,net,deltaOld= {},{},{}

def forward_propagation(l_dim,wb, inputs, update:bool = False):            
        in_out = inputs
        #forward propagation of the input pattern in the network
        for l in range(1, len(l_dim)):
            name_layer:str = str(l)
            w = wb[f'W{name_layer}'] # on the row there are the weights for 
            b = wb[f'b{name_layer}']
            #apply linear function
            linear_net = np.asarray(linear(w, in_out, b))
            #apply activation function to the net
            af = activations[C.SIGMOID]
            #traspose the result row unit column pattern
            in_out = np.asarray(af(linear_net).T)
            #if update true record the net and out of the units on the layer
            print(f'net{name_layer}:', linear_net)
            print(f'out{name_layer}:', in_out)
            if update:
                out["out0"] = inputs
                net[f'net{name_layer}'] = linear_net
                out[f'out{name_layer}'] = in_out
        return in_out
def sigmoid(net):
    
    """sigmoid_m=[]
    
    #avoid overflow
    for x in net.flatten():
        if x<0:
            sigmoid_m.append(np.exp(x)/(1+np.exp(x)))
        else:
            sigmoid_m.append(1 / (1 + np.exp(-x)))
            
    sigmoid_m=np.array(sigmoid_m)
    sigmoid_m=sigmoid_m.reshape(net.shape)"""
    return np.where(net >= 0, 1/(1 + np.exp(-net)), np.exp(net)/(1 + np.exp(net)))


def d_sigmoid(net):
    return np.multiply(sigmoid(net) , np.subtract( 1 , sigmoid(net)))

def loss( Y, Y_hat):
        l = 2 * Y.shape[0]
        dp = np.squeeze(Y) - np.squeeze(Y_hat)
        squares = np.square(dp)
        loss = np.sum(squares)
        loss = loss * (1/l)
        return loss

  
def d_loss( Y, Y_hat):
        #return (np.squeeze(Y) - np.squeeze(Y_hat)) * -1
        return (np.squeeze(Y) - np.squeeze(Y_hat))/len(Y_hat)

def compute_delta_k(y, out, net):
        #delta of loss function of pattern 
        dp = d_loss(y, out) #questa non mi torna
        #derivative of the layer activation function
        f_prime = d_sigmoid(net)
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
def compute_delta_j(delta_in, w_layer, net):
        # matrix multiplication for delta_t
        dt = delta_in @ w_layer
        # Transpose and puntual multiplication of apply the derivative
        fprime = d_sigmoid(net) # net is a vector with all the nets for the unit layer
        fprime = fprime if isinstance(fprime, int) else fprime.T
        # print(f'shape delta_in: {delta_in.shape}, w_layer: {w_layer.shape}, dt {dt.shape}')
        # input('premi')
        dj = dt * fprime
        return dj

def back_propagation(l_dim,wb,y):
        delta_t = []
        num_layers = len(l_dim) - 1
        #Reverse loop of the network layer
        for l in range(num_layers, 0, -1):
            dt = 0
            name_layer:str = str(l)
            # print(f'name_layer 228 ddn: {name_layer}')
            #recover the activation function of the layer
            if l != num_layers:
                dt = compute_delta_j(delta_t[-1], wb[f'W{l+1}'], net[f'net{name_layer}'])
            else:
                dt = compute_delta_k(y, out[f'out{name_layer}'], net[f'net{name_layer}'])
            #append the gradient matrix
            # print(f'dt.shape {dt.shape}, dt {dt}')
            # input('premi')
            delta_t.append(dt)
        return delta_t[::-1]

def update_wb(eta,l_dim ,delta, wb,pattern):
        p_eta:float = (eta)
        for l in range(len(l_dim) - 1, 0, -1):
            name_layer:str = str(l)

            deltaW =  (delta[l-1].T @out[f"out{l-1}"])  / pattern
            deltaB =  np.sum(delta[l-1].T, axis=1, keepdims=True)/ pattern

            
            deltaW = p_eta * deltaW
            deltaB = p_eta * deltaB
           
            wb[f'W{name_layer}']= wb[f'W{name_layer}'] + deltaW 
          
            wb[f'b{name_layer}'] = np.add(wb[f'b{name_layer}'] , deltaB)
          
        return wb
    
    
'''
    Compute metrics_binary_classification
    :param y: target values
    :param y_hat: predicted values
    :param treshold: treshold values
    :return result: dictionary with: C.ACCURACY,C.PRECISION,C.RECALL,C.SPECIFICITY,C.BALANCED
    '''
def metrics_binary_classification( y, y_hat, treshold = 0.5):
        if np.squeeze(y).shape != np.squeeze(y_hat).shape:
            raise Exception(f"Sets have different shape Y:{y.shape} Y_hat:{y_hat.shape}")

        tp,tn,fp,fn=0,0,0,0
        for predicted, target in zip(y_hat.flatten(),y.flatten()):
            if predicted < treshold:
                if target == 0:
                    tn+=1
                else:
                    fn+=1
            else:
                if target == 1:
                    tp+=1
                else:
                    fp+=1

        accuracy=(tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn >0 else 0
        recall= tp/(tp+fn) if tp+fn >0 else 0
        precision = tp/(tp+fp) if tp+fp >0 else 0
        specificity= tn/(tn+fp) if tn+fp >0 else 0
        balanced=0.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp  >0 else 0

        return {C.MISSCLASSIFIED: fp+fn,
        C.CLASSIFIED:tp+tn,
        C.ACCURACY:accuracy,
        C.PRECISION:precision,
        C.RECALL:recall,
        C.SPECIFICITY:specificity,
        C.BALANCED:balanced
        } 
        
import readMonk_and_Cup as readMC

def main():
    init_global()
    TR_x_monk1,TR_y_monk1 = readMC.get_train_Monk_1()
    TS_x_monk1,TS_y_monk1 = readMC.get_test_Monk_1()

    print("Inizializzo i pesi per una rete 2 - 2 - 1 - 1 ")
    l_dim=[17,2,1]
    inputx=TR_x_monk1
    targety=TR_y_monk1
    #inputx=np.array([[0,0],[0,0],[1,1],[1,1]])
    #targety=[0,0,1,1]
    # pattern=inputx.shape[0]
    print("Pattern")
    wb=init_wb(l_dim)
    num_layers:int = len(l_dim)
    for i in range(1000):
        for l in range(1, num_layers):
            name_layer:str = str(l)
            print("Layer",name_layer)
            print("W"+name_layer, wb[f'W{name_layer}'], wb[f'W{name_layer}'].shape)
            print("B"+name_layer, wb[f'b{name_layer}'],wb[f'b{name_layer}'].shape)
        
        in_out=forward_propagation(l_dim,wb,inputx,True)
        delta=back_propagation(l_dim,wb,targety)
        wb=update_wb(5,l_dim,delta,wb,1)
        if i % 500 == 0:
            print("Result:",metrics_binary_classification(targety,in_out))
            input(f"{i}: premi")
    in_out=forward_propagation(l_dim,wb,inputx,False)
    print("Result:",metrics_binary_classification(targety,in_out))
    input("Premi")
    in_out=forward_propagation(l_dim,wb,TS_x_monk1,False)
    print("Test Result:",metrics_binary_classification(TS_y_monk1,in_out))

if __name__ == "__main__":
    main()