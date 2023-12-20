from didacticNeuralNetwork import DidacticNeuralNetwork as dnn 
from kfoldCV import KfoldCV
import readMonk_and_Cup as readMC

"""if __name__ == '__main__':
    TR_x_monk2,TR_y_monk2 = readMC.get_train_Monk_2()
    kfCV = KfoldCV(TR_x_monk2, TR_y_monk2, 5)
    winner = kfCV.validate(FineGS = True)
    print(winner.to_string())

    # Hold-out Test 2
    model=dnn(**winner.get_dictionary())
    TS_x_monk2,TS_y_monk2= readMC.get_test_Monk_2()
    model.fit(TR_x_monk2,TR_y_monk2,TR_x_monk2,TR_y_monk2)
    out = model.forward_propagation(TS_x_monk2)
    error={}
    error['mean_absolute_error'] = model.metrics.mean_absolute_error(TS_y_monk2, out)
    error['root_mean_squared_error'] = model.metrics.root_mean_squared_error(TS_y_monk2, out)
    error['mean_euclidean_error'] = model.metrics.mean_euclidean_error(TS_y_monk2, out)
    print(error)"""
def main():
    TR_x_monk2,TR_y_monk2 = readMC.get_train_Monk_2()
    kfCV = KfoldCV(TR_x_monk2, TR_y_monk2, 5)
    winner = kfCV.validate(FineGS = True)
    print(winner.to_string())
    # Hold-out Test 1
    model=dnn(**winner.get_dictionary())
    TS_x_monk2,TS_y_monk2= readMC.get_test_Monk_2()
    error=model.fit(TR_x_monk2,TR_y_monk2,TS_x_monk2,TS_y_monk2)
    out = model.forward_propagation(TS_x_monk2)

    classification=model.metrics.metrics_binary_classification(TS_y_monk2,out)
    print("Test Accuracy:",classification["accuracy"],"Classified:",classification["classified"],"MisClassified:",classification["misclassified"])
if __name__ == '__main__':
    main()