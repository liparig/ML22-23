from didacticNeuralNetwork import DidacticNeuralNetwork as dnn
from kfoldCV import KfoldCV 
import readMonk_and_Cup as readMC

TR_x_monk1,TR_y_monk1 = readMC.get_train_Monk_1()
kfCV = KfoldCV(TR_x_monk1, TR_y_monk1, 2)
winner = kfCV.validate(FineGS = True)
print(winner.to_string())

# Hold-out Test 1
model=dnn(**winner.get_dictionary())
TS_x_monk1,TS_y_monk1= readMC.get_test_Monk_1()
model.fit(TR_x_monk1,TR_y_monk1,TR_x_monk1,TR_y_monk1)
out = model.forward_propagation(TS_x_monk1)

error={}
error['mean_absolute_error'] = model.metrics.mean_absolute_error(TS_y_monk1, out)
error['root_mean_squared_error'] = model.metrics.root_mean_squared_error(TS_y_monk1, out)
error['mean_euclidean_error'] = model.metrics.mean_euclidean_error(TS_y_monk1, out)
print(error)