
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import readMonk_and_Cup as readMC


TR_x_monk1,TR_y_monk1 = readMC.get_train_Monk_1()
TS_x_monk1,TS_y_monk1 = readMC.get_test_Monk_1()

TR_x_monk2,TR_y_monk2 = readMC.get_train_Monk_2()
TS_x_monk2,TS_y_monk2 = readMC.get_test_Monk_2()

TR_x_monk3,TR_y_monk3 = readMC.get_train_Monk_3()
TS_x_monk3,TS_y_monk3 = readMC.get_test_Monk_3()


X = torch.tensor(TR_x_monk1, dtype=torch.float32)
y = torch.tensor(TR_y_monk1, dtype=torch.float32).reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(17, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid())

print(model)

loss_fn = nn.MSELoss() # binary cross entropy
optimizer = optim.SGD(model.parameters(), lr=0.2)


n_epochs = 200
batch_size = 20
 
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
# compute accuracy
y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")
 
# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))