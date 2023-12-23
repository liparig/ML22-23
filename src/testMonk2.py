from didacticNeuralNetwork import DidacticNeuralNetwork as dnn 
from kfoldCV import KfoldCV
import readMonk_and_Cup as readMC
import costants as C

def main():
    TR_x_monk2,TR_y_monk2 = readMC.get_train_Monk_2()
    TS_x_monk2,TS_y_monk2= readMC.get_test_Monk_2()
    
    kfCV = KfoldCV(TR_x_monk2, TR_y_monk2, 2)
    
    theta_batch = {C.L_NET:[[17,4,1]],
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

    theta_mini = theta_batch.copy()
    theta_mini[C.L_ETA]=[0.1]
    theta_mini[C.L_TAU]=[(20, 0.001),(20 , 0.005)]
    theta_mini[C.L_DIMBATCH]=[20]
    theta_mini[C.L_MOMENTUM]= [(C.NESTEROV,0.2)]
    
    #region Batch
    winnerBatch = kfCV.validate(inTheta =  theta_batch, FineGS = True, prefixFilename="Monk2Batch_")
    print(winnerBatch.to_string())
    # Hold-out Test 1
    modelBatch = dnn(**winnerBatch.get_dictionary())
    #{'error':history_terror,'loss':history_tloss, 'metric_tr':metric_tr, 'metric_val':metric_val, 'validation':validation_error, 'c_metrics':c_metric, 'epochs':epoch + 1}  
    modelBatch.fit(TR_x_monk2,TR_y_monk2,TS_x_monk2,TS_y_monk2)
    out = modelBatch.forward_propagation(TS_x_monk2)

    classificationBatch = modelBatch.metrics.metrics_binary_classification(TS_y_monk2, out)
    print("Test Accuracy:", classificationBatch[C.ACCURACY], "Classified:", classificationBatch[C.CLASSIFIED], "Missclassified:", classificationBatch[C.MISSCLASSIFIED])
    #endregion Batch

    #region Minibatch
    print("Training MiniBatch:", theta_mini)
    winnerMini = kfCV.validate(inTheta = theta_mini, FineGS = True, prefixFilename  = "Monk2MiniBatch_")
    print(winnerMini.to_string())
    # Hold-out Test 1
    modelMini = dnn(**winnerMini.get_dictionary())
    #{'error':history_terror,'loss':history_tloss, 'metric_tr':metric_tr, 'metric_val':metric_val, 'validation':validation_error, 'c_metrics':c_metric, 'epochs':epoch + 1}  
    modelMini.fit(TR_x_monk2,TR_y_monk2,TS_x_monk2,TS_y_monk2)
    out = modelMini.forward_propagation(TS_x_monk2)

    classificationMiniBatch = modelMini.metrics.metrics_binary_classification(TS_y_monk2, out)
    print("Test Accuracy:", classificationMiniBatch[C.ACCURACY], "Classified:", classificationMiniBatch[C.CLASSIFIED], "Missclassified:", classificationMiniBatch[C.MISSCLASSIFIED])
    #endregion MiniBatch
    
if __name__ == "__main__":
    main()