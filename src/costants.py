#Types Activation functions
SIGMOID:str = 'sigmoid'
TANH:str = 'tanh'
RELU:str = 'relu'
LEAKYRELU:str = 'leakyRelu'
IDENTITY:str = 'identity'

MSE:str = 'mse'

#DataSet columns
NUM_PATTERN_X:str = 'p_tx'
INPUT_TRAINING:str = 'x_train'
OUTPUT_TRAINING:str = 'y_train'
INPUT_VALIDATION:str = 'x_val'
OUTPUT_VALIDATION:str = 'y_val'

FORMATTIMESTAMP:str = '%d%m%Y-%H%M'
#HYPERPARAMETERS
R_SEEDS:int = 22
EPS:float = 0.7 #interval for initialization of the weights
BIAS:float = 0 #w0
AFUNCTION:list = [] # activation functions list
BATCH:int = 0 #dimension of the batchs
ETA:float = 0.5 #learning rate
LMS:bool = True # dividing by L the gradients
TAU = (False, False) #learning decay rate (tau, eta_t) tau qualche centinaio di epoche eta_t di solito l'1% di eta iniziale
REG = (False, False) #regulation rate
EPOCHS:int = 1000 #Max Epochs
PATIENCE:int = 20 #Max Epochs with small changes
MOMENTUM = ("",0)
EARLYSTOP:bool = True
TRESHOLDVARIANCE:float = 1.e-12
BATCH_SHUFFLE=True

UNIFORM:str = 'uniform'
RANDOM:str = 'random'
BASIC: str = 'basic'

#Type Momentum
CLASSIC:str = 'classic'
NESTEROV:str = 'nesterov'

PATH_PLOT_DIR:str = "../plot/"
PATH_KFOLDCV_DIR:str="../KFoldCV"
PREFIX_DIR_COARSE:str = "Coarse"
PREFIX_DIR_FINE:str = "Fine"

LABEL_PLOT_TRAINING:str = "Training"
LABEL_PLOT_VALIDATION:str = "Validation"

TIKHONOV:str = 'tikhonov'
LASSO:str = 'lasso'

CUP:str = "cup"
MONK:str = "monk"

#Label HYPERPARAMETERS key dictionary
L_NET:str = 'l_dim'
L_ACTIVATION:str = 'a_functions'
L_ETA:str = 'eta'
L_TAU:str = 'tau'
L_REG:str = 'reg'
L_DIMBATCH:str = 'dim_batch'
L_MOMENTUM:str = 'momentum'
L_EPOCHS:str = 'epochs'
L_SHUFFLE:str = 'batch_shuffle'
L_EPS:str =  'eps'
L_DISTRIBUTION:str = 'distribution'
L_BIAS:str = 'bias'
L_SEED:str =  'seed'
L_CLASSIFICATION:str =  'classification'
L_EARLYSTOP:str = 'early_stop'
L_PATIENCE:str = 'patience'
L_TRESHOLD_VARIANCE:str = 'treshold_variance'


#Label metrics key dictionary
MISSCLASSIFIED:str = 'missclassified'
CLASSIFIED:str = 'classified'
ACCURACY:str = 'accuracy'
PRECISION:str = 'precision'
RECALL:str = 'recall'
SPECIFICITY:str = 'specificity'
BALANCED:str = 'balanced'

# type of dataser
TRAINING:str = 'training'
VALIDATION:str = 'validation'

# prefix file name and directorys
PREFIXBATCH:str = '_Batch_'
PREFIXMINIBATCH:str = '_Mini_'