from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV 
import readMonk_and_Cup as readMC
import dnn_plot
import costants as C
import time 

def main():
    TR_x_monk1,TR_y_monk1 = readMC.get_train_Monk_1()
    kfCV = KfoldCV(TR_x_monk1, TR_y_monk1, 5)
    winner = kfCV.validate(FineGS = False)
    print(winner.to_string())
    # Hold-out Test 1
    model=dnn(**winner.get_dictionary())
    TS_x_monk1,TS_y_monk1= readMC.get_test_Monk_1()
    error=model.fit(TR_x_monk1,TR_y_monk1,TS_x_monk1,TS_y_monk1)
    out = model.forward_propagation(TS_x_monk1)

    error['mean_absolute_error'] = model.metrics.mean_absolute_error(TS_y_monk1, out)
    error['root_mean_squared_error'] = model.metrics.root_mean_squared_error(TS_y_monk1, out)
    error['mean_euclidean_error'] = model.metrics.mean_euclidean_error(TS_y_monk1, out)
    classification=model.metrics.metrics_binary_classification(TS_y_monk1,out)
    plot_path =None
    inYlim = (-0.5, 1.5)
    dnn_plot.plot_curves(error['error'], error['validation'], error['metric_tr'], error['metric_val'], 
                        lbl_tr = C.LABEL_PLOT_TRAINING, lbl_vs = "Test", path = plot_path, 
                        ylim = inYlim, titleplot = f"Test Model assessment",
                        theta = winner.get_dictionary())
            
           
          
    print(classification["accuracy"],classification["classified"],classification["misclassified"])
    
if __name__ == "__main__":
    main()