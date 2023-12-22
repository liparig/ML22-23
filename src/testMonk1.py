from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV 
import readMonk_and_Cup as readMC
import dnn_plot
import costants as C

def main():
    TR_x_monk1,TR_y_monk1 = readMC.get_train_Monk_1()
    kfCV = KfoldCV(TR_x_monk1, TR_y_monk1, 2)
    theta_batch={C.L_NET:[[17,4,1]],
            C.L_ACTIVATION:[[C.TANH]],
            C.L_ETA:[0.6 , 0.5],
            C.L_TAU: [(500,0.06)],
            C.L_REG:[(False,False)],
            C.L_DIMBATCH:[0],
            C.L_MOMENTUM: [(False,False)],
            C.L_EPOCHS:[1000],
            C.L_SHUFFLE:True,
            C.L_EPS: [0.2 , 0.4],
            C.L_DISTRIBUTION:[C.UNIFORM],
            C.L_BIAS:[0],
            C.L_SEED: [52],
            C.L_CLASSIFICATION:True,
            C.L_EARLYSTOP:True,
            C.L_PATIENCE: [60],
            C.L_TRESHOLD_VARIANCE:[C.TRESHOLDVARIANCE]
        
    }
    theta_mini=theta_batch.copy()
    theta_mini[C.L_ETA]=[[0.3]]
    theta_mini[C.L_TAU]=[(30, 0.01),(0.30,0.05)]
    theta_mini[C.L_DIMBATCH]=[20]
    theta_mini[C.L_MOMENTUM]= [(C.NESTEROV,0.2)]
    
    winnerBatch = kfCV.validate(default="False",theta=theta_batch,FineGS = True,prefixFilename="Monk1Batch_")
    print(winnerBatch.to_string())
    # Hold-out Test 1
    modelBatch=dnn(**winnerBatch.get_dictionary())
    TS_x_monk1,TS_y_monk1= readMC.get_test_Monk_1()
    error=modelBatch.fit(TR_x_monk1,TR_y_monk1,TS_x_monk1,TS_y_monk1)
    out = modelBatch.forward_propagation(TS_x_monk1)

    classification=modelBatch.metrics.metrics_binary_classification(TS_y_monk1,out)
    print("Test Accuracy:",classification["accuracy"],"Classified:",classification["classified"],"MisClassified:",classification["misclassified"])
    
    
    winnerMini = kfCV.validate(default="False",theta=theta_batch,FineGS = True,prefixFilename="Monk1MiniBatch_")
    print(winnerMini.to_string())
    # Hold-out Test 1
    modelMini=dnn(**winnerMini.get_dictionary())
    TS_x_monk1,TS_y_monk1= readMC.get_test_Monk_1()
    error=modelMini.fit(TR_x_monk1,TR_y_monk1,TS_x_monk1,TS_y_monk1)
    out = modelMini.forward_propagation(TS_x_monk1)

    classification=modelMini.metrics.metrics_binary_classification(TS_y_monk1,out)
    print("Test Accuracy:",classification["accuracy"],"Classified:",classification["classified"],"MisClassified:",classification["misclassified"])
    
    
if __name__ == "__main__":
    main()