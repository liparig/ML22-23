#Types Activation functions
SIGMOID:str = 'sigmoid'
TANH:str = 'tanh'
RELU:str = 'relu'
IDENTITY:str = 'identity'

MSE:str = 'mse'

#DataSet columns
NUM_POINT_X:str = 'p_tx'
INPUT_TRAINING:str = 'x_train'
OUTPUT_TRAINING:str = 'y_train'
INPUT_VALIDATION:str = 'x_val'
OUTPUT_VALIDATION:str = 'y_val'

FORMATTIMESTAMP:str = '%d%m%Y-%H%M'

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
TRESHOLDVARIANCE:float = 1.e-6

UNIFORM:str = 'uniform'
RANDOM:str = 'random'
BASIC: str = 'basic'

#Type Momentum
CLASSIC:str = 'classic'
NESTEROV:str = 'nesterov'

PATH_PLOT_DIR:str = "../plot/"
PREFIX_DIR_COARSE:str = "Coarse"
PREFIX_DIR_FINE:str = "Fine"

LABEL_PLOT_TRAINING:str = "Training"
LABEL_PLOT_VALIDATION:str = "Validation"

TIKHONOV:str = 'tikhonov'
LASSO:str = 'lasso'