from costants import RELU, SIGMOID, TANH
from didacticNeuralNetwork import DidacticNeuralNetwork as dnn 
from kfoldCV import KfoldCV
import readMonk_and_Cup as readMC



 

 
 



TR_x_monk1,TR_y_monk1 = readMC.get_train_Monk_1()
TR_x_monk2,TR_y_monk2 = readMC.get_train_Monk_2()
TR_x_monk3,TR_y_monk3 = readMC.get_train_Monk_3()
# TS_x_monk1,TS_y_monk1 = readMC.get_test_Monk_1()
# TR_x_CUP,TR_y__CUP = readMC.get_train_CUP()
# TS_x_CUP,TS_y__CUP = readMC.get_test_CUP()
# tr_x,tr_y,val_x,val_y = readMC.get_cup_house_test()
# TR_x_monk1, TR_y_monk1, Vl_x_monk1, Vl_y_monk1 = readMC.read_monk_Tr_Vl()

# model = dnn(l_dim = [9,20,10,2], a_functions = [RELU, SIGMOID, 'identity'], early_stop = True, bias = 0, epochs = 1000, eta = 0.001, eps = 0.1, dim_batch = 0, classification = False, plot = "../plot")

# model = dnn(l_dim = [17,20,1], a_functions = [RELU, SIGMOID], early_stop = True, bias = 0, epochs = 1000, eta = 0.01, eps = 0.1, dim_batch = 0, classification = True, plot = "../plot")
# model = dnn(l_dim = [17,10,1], a_functions = [RELU, TANH], early_stop = True, bias = 0, epochs = 1000, eta = 0.01, eps = 0.1, dim_batch = 0, classification = True, plot = "../plot")

# error = model.fit(TR_x_monk1, TR_y_monk1, Vl_x_monk1, Vl_y_monk1)

# print(error['error'])
# print(error['validation'])
# error = model2.fit(tr_x,tr_y,val_x,val_y)
# error['epochs']

# kfCV= KfoldCV(TR_x_CUP,TR_y__CUP,2)
# winner=kfCV.validate(default="cup",FineGS=True)
# print(winner.to_string())

kfCV = KfoldCV(TR_x_monk1, TR_y_monk1, 5)
winner = kfCV.validate(FineGS = False)
print(winner.to_string())

#Hold-out Test
model=dnn(**winner.get_dictionary)
TS_x_monk1,TS_y_monk1= readMC.get_test_Monk_1()
model.fit(TR_x_monk1,TR_y_monk1,TR_x_monk1,TR_y_monk1)
out = model.forward_propagation(TS_x_monk1)
error=[]
error['mean_absolute_error'] = model.metrics.mean_absolute_error(TS_y_monk1, out)
error['root_mean_squared_error'] = model.metrics.root_mean_squared_error(TS_y_monk1, out)
error['mean_euclidean_error'] = model.metrics.mean_euclidean_error(TS_y_monk1, out)
print(error)
kfCV = KfoldCV(TR_x_monk2, TR_y_monk2, 5)
winner = kfCV.validate(FineGS = True)
print(winner.to_string())

kfCV = KfoldCV(TR_x_monk3, TR_y_monk3, 5)
winner = kfCV.validate(FineGS = True)
print(winner.to_string())