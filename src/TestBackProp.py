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

def compute_delta_k(y, out, net, d_activation):
        #delta of loss function of pattern 
        ms=MSE()
        dp = ms.d_loss(y, out) #questa non mi torna
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
def compute_delta_j(delta_in, w_layer, net, d_activation):
        # matrix multiplication for delta_t
        dt = delta_in @ w_layer
        # Transpose and puntual multiplication of apply the derivative
        fprime = d_activation(net) # net is a vector with all the nets for the unit layer
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
            af = C.SIGMOID
            #if not the output layer compute gradients delta_j
            if l != num_layers:
                dt = compute_delta_j(delta_t[-1], wb[f'W{l+1}'], net[f'net{name_layer}'], derivatives[af])
            else:
                dt = compute_delta_k(y, out[f'out{name_layer}'], net[f'net{name_layer}'], derivatives[af])
            #append the gradient matrix
            # print(f'dt.shape {dt.shape}, dt {dt}')
            # input('premi')
            delta_t.append(dt)
        return delta_t[::-1]

def update_wb(eta,l_dim ,delta, wb):
        p_eta:float = (eta)
        for l in range(len(l_dim) - 1, 0, -1):
            name_layer:str = str(l)
            print(f"Name {name_layer}" ,delta[l-1].T.shape,out[f"out{l-1}"].shape)
            deltaW = -p_eta * (delta[l-1].T@out[f"out{l-1}"])
            deltaB = -p_eta * (delta[l-1])
            
            print(f"deltaB",deltaB, f"delta{name_layer}-1",delta[l-1])
            #if regularization is set subtract the penalty term
            """if self.regular:
                deltaW -= self.regular.derivative(self, self.lambdar, self.wb[f'W{name_layer}'])
                deltaB -= self.regular.derivative(self, self.lambdar, self.wb[f'b{name_layer}'])
            #if momentum classic is set add the momentum deltaW
            if self.momentum == C.CLASSIC and f'wold{name_layer}' in self.deltaOld:
                deltaW += self.alpha*self.deltaOld[f'wold{name_layer}']
                deltaB += self.alpha*self.deltaOld[f'bold{name_layer}']"""

            # print(type(self.wb[f'W{name_layer}']))
            # print(type(deltaW))
            #print(f"B of {name_layer}",self.wb[f'b{name_layer}'])
            # print(self.wb[f'b{name_layer}'])
           # print(f"B {name_layer}",self.wb[f'b{name_layer}'])
           # print("Delta B",deltaB, deltaB.shape)
            print("PRIMA W"+name_layer,wb[f'W{name_layer}'].shape,"deltaW",deltaW.shape)
            wb[f'W{name_layer}']= wb[f'W{name_layer}'] + deltaW 
            print("Dopo W"+name_layer,wb[f'W{name_layer}'].shape,  "deltaW:",deltaW.shape )
            print("PRIMA B"+name_layer,wb[f'b{name_layer}'].shape,"deltab",deltaB.shape)
            wb[f'b{name_layer}'] = np.add(wb[f'b{name_layer}'] , deltaB.T)
            print("Dopo B"+name_layer,wb[f'b{name_layer}'].shape,  "deltab:",deltaB.shape )
            #print("B dopo upadate {name_layer}",self.wb[f'b{name_layer}'])
           # input('premi')

            # print(f"W:{self.wb[f'W{name_layer}']}, b:{self.wb[f'b{name_layer}']}")
            # input('premi')
            #save the old gradient for the momentum if needed
            deltaOld[f'wold{name_layer}'] = deltaW
            deltaOld[f'bold{name_layer}'] = deltaB
        return wb

def main():
    init_global()
    print("Inizializzo i pesi per una rete 2 - 2 - 1 ")
    wb=init_wb([2,2,1])
    num_layers:int = len([2,2,1])
    for l in range(1, num_layers):
        name_layer:str = str(l)
        print("Layer",name_layer)
        print("W"+name_layer, wb[f'W{name_layer}'], wb[f'W{name_layer}'].shape)
        print("B"+name_layer, wb[f'b{name_layer}'],wb[f'b{name_layer}'].shape)
    
    in_out=forward_propagation([2,2,1],wb,np.array([[2,1]]),True)
    print("Uscita rete",in_out)
    delta=back_propagation([2,2,1],wb,[1])
    print("Delta",delta)
    wb=update_wb(1,[2,2,1],delta,wb)
    print(wb)
    in_out=forward_propagation([2,2,1],wb,np.array([[2,1]]),True)

if __name__ == "__main__":
    main()