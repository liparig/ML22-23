from costants import RELU, SIGMOID, TANH
from didacticNeuralNetwork import DidacticNeuralNetwork as dnn 
from kfoldCV import KfoldCV
import readMonk_and_Cup as readMC

# TR_x_monk1,TR_y_monk1 = readMC.get_train_Monk_1()
# TS_x_monk1,TS_y_monk1 = readMC.get_test_Monk_1()
# TR_x_CUP,TR_y__CUP = readMC.get_train_CUP()
# TS_x_CUP,TS_y__CUP = readMC.get_test_CUP()
# tr_x,tr_y,val_x,val_y = readMC.get_cup_house_test()
TR_x_monk1, TR_y_monk1, Vl_x_monk1, Vl_y_monk1 = readMC.read_monk_Tr_Vl()
# print('TR_x_monk1')
# print(TR_x_monk1.shape)
# print(TR_x_monk1)
# input('premi')
# print('TR_y_monk1')
# print(TR_y_monk1.shape)
# print(TR_y_monk1)
# input('premi')
# print('Vl_x_monk1')
# print(Vl_x_monk1.shape)
# print(Vl_x_monk1)
# input('premi')
# print('Vl_y_monk1')
# print(Vl_y_monk1.shape)
# print(Vl_y_monk1)
# input('premi')
# model = dnn(l_dim = [9,20,10,2], a_functions = [RELU, SIGMOID, 'identity'], early_stop = True, bias = 0, epochs = 1000, eta = 0.001, eps = 0.1, dim_batch = 0, classification = False, plot = "../plot")

# model = dnn(l_dim = [17,20,1], a_functions = [RELU, SIGMOID], early_stop = True, bias = 0, epochs = 1000, eta = 0.01, eps = 0.1, dim_batch = 0, classification = True, plot = "../plot")
model = dnn(l_dim = [17,10,1], a_functions = [RELU, TANH], early_stop = True, bias = 0, epochs = 1000, eta = 0.01, eps = 0.1, dim_batch = 0, classification = True, plot = "../plot")

error = model.fit(TR_x_monk1, TR_y_monk1, Vl_x_monk1, Vl_y_monk1)

print(error['error'])
print(error['validation'])
# error = model2.fit(tr_x,tr_y,val_x,val_y)
# error['epochs']

# kfCV= KfoldCV(TR_x_CUP,TR_y__CUP,2)
# winner=kfCV.validate(default="cup",FineGS=True)
# print(winner.to_string())

# kfCV = KfoldCV(TR_x_monk1, TR_y_monk1, 2)
# winner = kfCV.validate()
# print(winner.to_string())