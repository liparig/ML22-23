from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV 
import readMonk_and_Cup as readMC
import dnn_plot
import costants as C

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

    classification=model.metrics.metrics_binary_classification(TS_y_monk1,out)
    print("Test Accuracy:",classification["accuracy"],"Classified:",classification["classified"],"MisClassified:",classification["misclassified"])
    
if __name__ == "__main__":
    main()