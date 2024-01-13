import time
import costants as C
from cupEvaluation import cup_evaluation, ensemble_Cup
import readMonkAndCup as readMC

#it's a test for get the performance with test dataset
# the methos are in the cupEvaluation.py file

def main(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, dirName):
    # theta_batch = {
    #     C.L_NET:[[10,15,5,3]],
    #     C.L_ACTIVATION:[[C.RELU,C.RELU,C.IDENTITY],[C.TANH,C.IDENTITY],[C.LEAKYRELU,C.IDENTITY],[C.LEAKYRELU,C.LEAKYRELU,C.IDENTITY],[C.SIGMOID,C.TANH,C.IDENTITY]],
    #     C.L_ETA:[0.03, 0.1],
    #     C.L_TAU: [(200,0.003), (False,False)],
    #     C.L_REG:[(C.LASSO,0.01),(C.TIKHONOV,0.01), (False,False)],
    #     C.L_DIMBATCH:[0],
    #     C.L_MOMENTUM: [(C.CLASSIC,0.9), (False,False)],
    #     C.L_EPOCHS:[500],
    #     C.L_SHUFFLE:True,
    #     C.L_EPS: [0.01],
    #     C.L_DISTRIBUTION:[C.GLOROT],
    #     C.L_G_CLIPPING:[(True,5)],
    #     C.L_DROPOUT:[C.DROPOUT],
    #     C.L_BIAS:[0],
    #     C.L_SEED: [52],
    #     C.L_CLASSIFICATION:False,
    #     C.L_EARLYSTOP:True,
    #     C.L_PATIENCE: [10],
    #     C.L_TRESHOLD_VARIANCE:[1.e-2]    
    # }
    theta_batch = {
        C.L_NET:[[10,8,5,3],[10,15,3]],
        C.L_ACTIVATION:[[C.TANH,C.IDENTITY],[C.LEAKYRELU,C.IDENTITY],[C.LEAKYRELU,C.LEAKYRELU,C.IDENTITY],[C.TANH,C.TANH,C.IDENTITY]],
        C.L_ETA:[0.004,0.008],
        C.L_TAU: [(False,False),(500,0.0008)],
        C.L_REG:[(C.TIKHONOV,0.002),(C.LASSO,0.002),(False,False)],
        C.L_DIMBATCH:[0],
        C.L_MOMENTUM: [(C.CLASSIC, 0.7),(C.NESTEROV, 0.7)],
        C.L_EPOCHS:[2000],
        C.L_SHUFFLE:True,
        C.L_EPS: [0.01],
        C.L_DISTRIBUTION:[C.GLOROT],
        C.L_G_CLIPPING:[(True,7),C.G_CLIPPING],
        C.L_DROPOUT:[C.DROPOUT],
        C.L_BIAS:[0],
        C.L_SEED: [52],
        C.L_CLASSIFICATION:False,
        C.L_EARLYSTOP:True,
        C.L_PATIENCE: [10],
        C.L_TRESHOLD_VARIANCE:[1.e-10]    
    }

    #batchWinners_list = cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, theta_batch, dirName, prefixFilename = C.PREFIXBATCH, fold = 5)
    models = []
    #models.extend(batchWinners_list)
    """theta_mini = {
        C.L_NET:[[10,20,3],[10,15,5,3]],
        C.L_ACTIVATION:[[C.RELU,C.RELU,C.IDENTITY],[C.TANH,C.IDENTITY]],
        C.L_ETA:[0.00003],
        C.L_TAU: [(20,0.0003)],
        C.L_REG:[(C.LASSO,0.00001),(C.TIKHONOV,0.00001)],
        C.L_DIMBATCH:[1,50,100],
        C.L_MOMENTUM: [(C.CLASSIC,0.6),(C.NESTEROV,0.6)],
        C.L_EPOCHS:[500],
        C.L_SHUFFLE:True,
        C.L_EPS: [0.001],
        C.L_DISTRIBUTION:[C.GLOROT],
        C.L_BIAS:[0],
        C.L_SEED: [52],
        C.L_CLASSIFICATION:False,
        C.L_EARLYSTOP:True,
        C.L_PATIENCE: [10],
        C.L_TRESHOLD_VARIANCE:[1.e-1]    
    }"""
    theta_mini = {
        C.L_NET:[[10,15,3],[10,8,8,8,3]],
        C.L_ACTIVATION:[[C.LEAKYRELU,C.LEAKYRELU,C.LEAKYRELU,C.IDENTITY],[C.LEAKYRELU,C.IDENTITY]],
        C.L_ETA:[0.005,0.05],
        C.L_TAU: [(500,0.005),(False,False)],
        C.L_REG:[(C.TIKHONOV,0.001),(False,False)],
        C.L_DIMBATCH:[100,50,150],
        C.L_MOMENTUM: [(C.CLASSIC,0.7)],
        C.L_EPOCHS:[400,500],
        C.L_SHUFFLE:True,
        C.L_EPS: [0.001],
        C.L_DISTRIBUTION:[C.GLOROT],
        C.L_G_CLIPPING:[C.G_CLIPPING,(True,10)],
        C.L_DROPOUT:[C.DROPOUT,(True,0.7)],
        C.L_BIAS:[0],
        C.L_SEED: [52],
        C.L_CLASSIFICATION:False,
        C.L_EARLYSTOP:False,
        C.L_PATIENCE: [10],
        C.L_TRESHOLD_VARIANCE:[1.e-5]    
    }

    minibatchWinners_list = cup_evaluation(inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, theta_mini, dirName, prefixFilename = C.PREFIXMINIBATCH, fold = 5)
    models.extend(minibatchWinners_list)
    ensemble_Cup(models, inTR_x_cup, inTR_y_cup, inTS_x_cup, inTS_y_cup, dirName, filename = "Ensemble_")
    
if __name__ == "__main__":
    start = time.time()
    TR_x_CUP, TR_y_CUP,  TS_x_CUP, TS_y_CUP = readMC.get_cup_house_test()
    DIRNAME:str = "TestCUP_1"
    # print(TR_x_CUP.shape, TR_y_CUP.shape,  TS_x_CUP.shape, TS_y_CUP.shape)
    # print(TR_x_CUP[0], TR_y_CUP[0],  TS_x_CUP[0], TS_y_CUP[0])
    # input('premi')
    main(TR_x_CUP, TR_y_CUP, TS_x_CUP, TS_y_CUP, DIRNAME)
    end = time.time()

    print(f'Cup Test takes {(end - start)/3600} hours')