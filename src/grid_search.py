#from validation.distribuited_computing import kfold_distributed_computing_cup
from candidate_hyperparameters import Candidates_Hyperparameters
from candidate_hyperparameters import Candidate


def init_grid_search(candidates, coarse):
    """ This function fills the set of possibile hyperparameters for the Grid Search
        in two possible ways, for a coarse grid search and for a finer one
    """
    global possibles_eta, possibles_momentum, possibles_reg, possibles_dim_batch, possibles_l_dim, possibles_a_functions,\
    possibles_tau,possibles_eps,possibles_distribution,possibles_bias,possibles_patience,classification
   
    possibles_l_dim          = candidates.l_dim
    possibles_a_functions    = candidates.a_functions
    possibles_distribution   = candidates.distribution
    possibles_patience       = candidates.patience
    possibles_bias           = candidates.bias
    classification           = candidates.classification  

    # Default values for Coarse Grid Search (values differ in order of magnitude)
    if coarse:
        possibles_eta           = candidates.eta
        possibles_momentum      = candidates.momentum
        possibles_reg           = candidates.reg
        possibles_dim_batch     = candidates.dim_batch
        possibles_tau           = candidates.tau
        possibles_eps           = candidates.eps
     # Edited values for Fine Grid, taken in a small range of the winner values of the Coarse Grid Search
    else:
        possibles_eta           = candidates.get_fine_range(candidates.eta)
        possibles_momentum      = candidates.get_fine_tuple(candidates.momentum)
        possibles_reg           = candidates.get_fine_tuple(candidates.reg)
        possibles_tau           = candidates.get_fine_tuple(candidates.tau)
        possibles_eps           = candidates.get_fine_range(candidates.eps)
        possibles_dim_batch     = candidates.get_fine_batch_size(candidates.dim_batch)

def grid_search(hyperparameters, coarse):
    candidates = Candidates_Hyperparameters()
    """ Grid Search
        :param : context, a reference of the KFold class
        :param : hyperparameters, in the case of Coarse grid search, indicates the list of possible hyperparameters to be searched.
                                  in the case of Fine grid search, indicates the best hyperparameter (which won the Coarse grid search)
        :param : coarse, a Boolean value
        :param : callback_for_each_hyperparameter, a callback which is called in order to calculate its folder
        :param : final_callback, a callback which is called at the end of the GS
        :param : stopping_criteria, an object for stopping criteria
    """
    init_grid_search(hyperparameters, coarse)
    count = 0
  
    if coarse:
        permutation = len(possibles_eta) * len(possibles_momentum)\
         * len(possibles_tau)  * len(possibles_dim_batch)\
         * len(possibles_reg) * len(possibles_eps) * len(possibles_l_dim)\
         * len(possibles_distribution) *  len(possibles_bias)\
         * len(possibles_a_functions) *  len(possibles_patience)
    if coarse:
        """ cycle over all the permutation values of hyperparameters """
        for eta in possibles_eta:
            for momentum in possibles_momentum:
                for tau in possibles_tau:
                    for dim_batch in possibles_dim_batch:
                            for l_dim in possibles_l_dim:
                                for a_functions in possibles_a_functions:
                                    if len(a_functions) != 1 and len(a_functions) != len(l_dim)-1:
                                        count += 1
                                    else:
                                        for eps in possibles_eps:
                                            for distribution in possibles_distribution:
                                                for patience in possibles_patience:
                                                    for bias in possibles_bias:
                                                        for reg in possibles_reg:
                                                            if count == 0 or count%100 == 0 or count == permutation-1:
                                                                print("Create the candidate:", count+1, "/", permutation)
                                                            count += 1
                                                            candidates.insert_candidate(l_dim=l_dim, a_functions=a_functions, eta=eta, tau=tau, reg=reg,\
            dim_batch=dim_batch, momentum=momentum,eps=eps,distribution=distribution,\
            bias=bias, classification=classification,patience=patience)
    else:
        """ cycle over all the permutation values of hyperparameters """
        permutation = len(possibles_eta) * len(possibles_momentum)* len(possibles_tau)  * len(possibles_dim_batch) * len(possibles_eps) * len(possibles_reg)
        for eta in possibles_eta:
            for momentum in possibles_momentum:
                for tau in possibles_tau:
                    for dim_batch in possibles_dim_batch:
                        for eps in possibles_eps:
                            for reg in possibles_reg:
                                if count == 0 or count%10000 == 0 or count == permutation-1:
                                    print("Create the candidate:",count+1,"/", permutation)
                                count += 1
                                candidates.insert_candidate(l_dim=hyperparameters.l_dim, a_functions=hyperparameters.a_functions, eta=eta, tau=tau, reg=reg,\
            dim_batch=dim_batch, momentum=momentum,eps=eps,distribution=hyperparameters.distribution,\
            bias=hyperparameters.bias, classification=hyperparameters.classification,patience=hyperparameters.patience)

    return candidates, count
