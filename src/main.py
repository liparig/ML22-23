from didacticNeuralNetwork import DidacticNeuralNetwork as dnn 
from kfoldCV import KfoldCV
import readMonk_and_Cup as readMC

# TR_x_monk1,TR_y_monk1 = readMC.get_train_Monk_1()
# TS_x_monk1,TS_y_monk1 = readMC.get_test_Monk_1()
# TR_x_CUP,TR_y__CUP = readMC.get_train_CUP()
# TS_x_CUP,TS_y__CUP = readMC.get_test_CUP()
# tr_x,tr_y,val_x,val_y = readMC.get_cup_house_test()
TR_x_monk1, TR_y_monk1, Vl_x_monk1, Vl_y_monk1 = readMC.read_monk_Tr_Vl()

# model = dnn(l_dim = [9,20,10,2], a_functions = ['relu','sigmoid','identity'], early_stop = True, bias = 0, epochs = 1000, eta = 0.001, eps = 0.1, dim_batch = 0, classification = False, plot = "../plot")

model = dnn(l_dim = [6,10,10,1], a_functions = ['relu', 'sigmoid', 'sigmoid'], early_stop = True, bias = 0, epochs = 1000, eta = 0.001, eps = 0.1, dim_batch = 0, classification = True, plot = "../plot")

error = model.fit(TR_x_monk1, TR_y_monk1, Vl_x_monk1, Vl_y_monk1)
print(error)

# error = model2.fit(tr_x,tr_y,val_x,val_y)
# error['epochs']

# kfCV= KfoldCV(TR_x_CUP,TR_y__CUP,2)
# winner=kfCV.validate(default="cup",FineGS=True)
# print(winner.to_string())

# kfCV = KfoldCV(TR_x_monk1, TR_y_monk1, 2)
# winner = kfCV.validate()
# print(winner.to_string())