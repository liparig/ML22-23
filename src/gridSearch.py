#from validation.distribuited_computing import kfold_distributed_computing_cup
from candidateHyperparameters import CandidatesHyperparameters
from candidateHyperparameters import Candidate

# This function fills the set of possibile hyperparameters for the Grid Search
# in two possible ways, for a coarse grid search and for a finer one
# :param: candidates are the candidate hyperparameters
# :param: coarse is a flag for indicate if is a coarse grid search
def init_grid_search(candidates:CandidatesHyperparameters|Candidate, coarse:bool):
    
    global possibles_eta, possibles_momentum, possibles_reg, possibles_dim_batch, possibles_l_dim, possibles_a_functions,\
    possibles_tau, possibles_eps, possibles_distribution, possibles_bias, possibles_patience, classification, early_stop,\
    possibles_epochs, possibles_threshold_variance,seed,possibles_g_clipping,possibles_dropout

    possibles_l_dim              = candidates.l_dim
    possibles_a_functions        = candidates.a_functions
    possibles_distribution       = candidates.distribution
    possibles_patience           = candidates.patience
    possibles_bias               = candidates.bias
    classification               = candidates.classification  
    early_stop                   = candidates.early_stop
    seed                         = candidates.seed
    possibles_threshold_variance = candidates.treshold_variance
    possibles_dropout            = candidates.dropout
    
    # Default values for Coarse Grid Search (values differ in order of magnitude)
    if coarse:
        possibles_eta           = candidates.eta
        possibles_momentum      = candidates.momentum
        possibles_reg           = candidates.reg
        possibles_dim_batch     = candidates.dim_batch
        possibles_tau           = candidates.tau
        possibles_eps           = candidates.eps
        possibles_epochs        = candidates.epochs
        possibles_g_clipping    = candidates.g_clipping
        
        

    # Edited values for Fine Grid, taken in a small range of the winner values of the Coarse Grid Search
    else:
        possibles_eta           = candidates.get_fine_range(candidates.eta)
        possibles_momentum      = candidates.get_fine_tuple(candidates.momentum)
        possibles_reg           = candidates.get_fine_tuple(candidates.reg)
        possibles_g_clipping    = candidates.get_fine_tuple(candidates.g_clipping)
        possibles_tau           = candidates.get_fine_tuple(candidates.tau)
        possibles_eps           = candidates.get_fine_range(candidates.eps)
        possibles_dim_batch     = candidates.get_fine_batch_size(candidates.dim_batch)
        possibles_epochs        = candidates.get_fine_int_range(candidates.epochs)
   
# Grid Search
# :param : hyperparameters, in the case of Coarse grid search, indicates the list of possible hyperparameters to be searched.
# in the case of Fine grid search, indicates the best hyperparameter (which won the Coarse grid search)
# :param : coarse, a Boolean value
def grid_search(hyperparameters:CandidatesHyperparameters|Candidate, coarse:bool = True):
    candidates = CandidatesHyperparameters()
    
    init_grid_search(hyperparameters, coarse)
    
    count:int = 0
    if coarse:
        permutation:int = len(possibles_l_dim) * len(possibles_a_functions) * len(possibles_eta)\
         * len(possibles_momentum) * len(possibles_reg) * len(possibles_dim_batch)\
         * len(possibles_tau) * len(possibles_patience) * len(possibles_eps)\
         * len(possibles_distribution) *  len(possibles_bias)\
         * len(possibles_epochs) * len(possibles_threshold_variance) *len(possibles_g_clipping) *len(possibles_dropout)
        
        """ cycle over all the permutation values of hyperparameters """
        for eta in possibles_eta:
            for momentum in possibles_momentum:
                for tau in possibles_tau:
                    for dim_batch in possibles_dim_batch:
                        for l_dim in possibles_l_dim:
                            for a_functions in possibles_a_functions:
                                # if there are some problems with the configuration dimensions of the layer or the activation functions
                                if len(a_functions) != 1 and len(a_functions) != len(l_dim) - 1:
                                    permutation -= 1
                                else:
                                    for eps in possibles_eps:
                                        for distribution in possibles_distribution:                                                    
                                            for patience in possibles_patience:
                                                for bias in possibles_bias:
                                                    for reg in possibles_reg:
                                                        for g_clipping in possibles_g_clipping:
                                                            for dropout in possibles_dropout:
                                                                for epochs in possibles_epochs:
                                                                    for treshold_variance in possibles_threshold_variance:
                                                                        if count == 0 or count%100 == 0 or count == permutation-1:
                                                                            print("Create the candidate:", count+1, "/", permutation)
                                                                        count += 1
                                                                        candidates.insert_candidate(l_dim=l_dim, a_functions=a_functions, eta=eta, tau=tau,g_clipping=g_clipping, dropout=dropout, reg=reg,\
                                                                                dim_batch=dim_batch, momentum=momentum,eps=eps,distribution=distribution,\
                                                                                bias=bias, classification=classification,patience=patience,early_stop=early_stop,epochs=epochs,seed=seed,treshold_variance=treshold_variance)

    else:
        """ cycle over all the permutation values of hyperparameters """
        permutation:int = len(possibles_eta) * len(possibles_momentum)* len(possibles_tau)  * len(possibles_dim_batch) * len(possibles_eps)  *len(possibles_g_clipping)
        for eta in possibles_eta:
            for momentum in possibles_momentum:
                for tau in possibles_tau:
                    for dim_batch in possibles_dim_batch:
                        for eps in possibles_eps:
                            for reg in possibles_reg:
                                for g_clipping in possibles_g_clipping:
                                    if count == 0 or count%10000 == 0 or count == permutation-1:
                                        print("Create the candidate:",count+1,"/", permutation)
                                    count += 1
                                    candidates.insert_candidate(l_dim=hyperparameters.l_dim, a_functions=hyperparameters.a_functions, eta=eta, tau=tau, g_clipping=g_clipping, dropout= hyperparameters.dropout, reg=reg,\
                dim_batch = dim_batch, momentum=momentum,eps=eps,distribution=hyperparameters.distribution,\
                bias = hyperparameters.bias, classification=hyperparameters.classification,patience=hyperparameters.patience,\
                    epochs=hyperparameters.epochs ,early_stop=hyperparameters.early_stop,seed=hyperparameters.seed,treshold_variance=hyperparameters.treshold_variance)
                                
    return candidates, count
