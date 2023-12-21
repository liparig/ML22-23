from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV 
import readMonk_and_Cup as readMC
import costants as C

    
def main():
    TR_x_CUP, TR_y_CUP,  TS_x_CUP, TS_y_CUP = readMC.get_cup_house_test()
    print(TR_x_CUP)
    kfCV = KfoldCV(TR_x_CUP, TR_y_CUP, 5)
    winner = kfCV.validate(default=C.CUP,FineGS = True)
    print(winner.to_string())

    # Hold-out Test 1
    model=dnn(**winner.get_dictionary())
    #TS_x_CUP,TS_y_CUP= readMC.get_test_CUP()
    error=model.fit(TR_x_CUP,TR_y_CUP,TS_x_CUP,TS_y_CUP)
    out = model.forward_propagation(TS_x_CUP)

    regression=model.metrics.mean_euclidean_error(TS_y_CUP,out)
    print("Test MEE:",regression)

if __name__ == '__main__':
    main()