from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV 
import readMonk_and_Cup as readMC
import costants as C
from dnn_plot import plot_curves 
import os

def main():
    TR_x_monk1,TR_y_monk1 = readMC.get_train_Monk_1()
    TS_x_monk1,TS_y_monk1= readMC.get_test_Monk_1()
    
    kfCV = KfoldCV(TR_x_monk1, TR_y_monk1, 2)
    
    theta_batch = {C.L_NET:[[17,3,1]],
            C.L_ACTIVATION:[[C.TANH]],
            C.L_ETA:[0.8 ,0.5],
            C.L_TAU: [(500,0.2)],
            C.L_REG:[(False,False),(C.TIKHONOV,0.01)],
            C.L_DIMBATCH:[0],
            C.L_MOMENTUM: [(False,False)],
            C.L_EPOCHS:[2000],
            C.L_SHUFFLE:True,
            C.L_EPS: [0.2],
            C.L_DISTRIBUTION:[C.UNIFORM],
            C.L_BIAS:[0],
            C.L_SEED: [52],
            C.L_CLASSIFICATION:True,
            C.L_EARLYSTOP:True,
            C.L_PATIENCE: [20,50],
            C.L_TRESHOLD_VARIANCE:[C.TRESHOLDVARIANCE]
        
    }

    theta_mini = theta_batch.copy()
    theta_mini[C.L_ETA]=[0.07]
    theta_mini[C.L_TAU]=[(200, 0.007),(300 , 0.005)]
    theta_mini[C.L_DIMBATCH]=[25]
    theta_mini[C.L_MOMENTUM]= [(C.NESTEROV,0.8),(C.CLASSIC,0.8)]
    
    winnerBatch = kfCV.validate(inTheta =  theta_batch, FineGS = True, prefixFilename="Monk1Batch_")
    # Hold-out Test 1
    modelBatch = dnn(**winnerBatch.get_dictionary())
    error=modelBatch.fit(TR_x_monk1,TR_y_monk1,TS_x_monk1,TS_y_monk1)
    out = modelBatch.forward_propagation(TS_x_monk1)

    classificationBatch = modelBatch.metrics.metrics_binary_classification(TS_y_monk1, out)
    labelMetric = C.ACCURACY

    # Path
    path_dir_models_coarse = os.path.join(C.PATH_PLOT_DIR, "Test Monk1")
    if not os.path.exists(path_dir_models_coarse):
            os.makedirs(path_dir_models_coarse)

    plot_curves(error['error'], error['validation'], error['metric_tr'], error['metric_val'], error_tr=error['loss'],
                        lbl_tr = C.LABEL_PLOT_TRAINING, lbl_vs = C.LABEL_PLOT_VALIDATION, path = path_dir_models_coarse+"/TestBatchMonk1", 
                        ylim = (-0.5, 1.5), titleplot = f"Test Model",
                        theta = winnerBatch.get_dictionary(), labelsY = ['Loss', labelMetric])

    print("Test Accuracy:", classificationBatch[C.ACCURACY], "Classified:", classificationBatch[C.CLASSIFIED], "MissClassified:", classificationBatch[C.MISSCLASSIFIED])

    #region Minibatch
    print("Training MiniBatch:", theta_mini)
    winnerMini = kfCV.validate(inTheta = theta_mini, FineGS = True, prefixFilename = "Monk1MiniBatch_")
    # Hold-out Test 1
    modelMini = dnn(**winnerMini.get_dictionary())
    #{'error':history_terror,'loss':history_tloss, 'metric_tr':metric_tr, 'metric_val':metric_val, 'validation':validation_error, 'c_metrics':c_metric, 'epochs':epoch + 1}  
    error = modelMini.fit(TR_x_monk1,TR_y_monk1,TS_x_monk1,TS_y_monk1)
    out = modelMini.forward_propagation(TS_x_monk1)

    classificationMiniBatch = modelMini.metrics.metrics_binary_classification(TS_y_monk1, out)
    print("Test Accuracy:", classificationMiniBatch[C.ACCURACY], "Classified:", classificationMiniBatch[C.CLASSIFIED], "Missclassified:", classificationMiniBatch[C.MISSCLASSIFIED])
    #endregion MiniBatch

    plot_curves(error['error'], error['validation'], error['metric_tr'], error['metric_val'], error_tr=error['loss'],
                        lbl_tr = C.LABEL_PLOT_TRAINING, lbl_vs = C.LABEL_PLOT_VALIDATION, path =  path_dir_models_coarse+"/TestMiniBatchMonk1", 
                        ylim = (-0.5, 1.5), titleplot = f"Test Model",
                        theta = winnerMini.get_dictionary(), labelsY = ['Loss', labelMetric])
    
if __name__ == "__main__":
    main()