from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV 
import readMonk_and_Cup as readMC
import dnn_plot
import costants as C
import time 

def main():
    TR_x_CUP,TR_y_CUP = readMC.get_train_CUP()
    kfCV = KfoldCV(TR_x_CUP, TR_y_CUP, 2)
    winner = kfCV.validate(default="cup",FineGS = False)
    print(winner.to_string())

    
if __name__ == "__main__":
    main()