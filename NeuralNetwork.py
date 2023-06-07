import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import csv

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
#load only from column 0 to column 8

X = dataset[0:614,0:8]

#load column 8 which is the output vector
y = dataset[0:614,8]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

#The input layer has 8 nodes corresponding to each feature of the data.
#The first layer has 12 neurons and the activation function is ReLU
#The second layer has 8 neurons and the activation function is ReLU
#The ouput layer has only one node as it is a binary classification problem
#nn.Linear will compute g(W(l)*A(l-1) + b(l)) where W(l) is the weight matrix of the lth layer.
#The dimensions of W(l) = n(l)* n(l-1). In the first layer, W will be (12*8), X =  (8 * n), b = (12 * 1)
model = nn.Sequential(
    nn.Linear(8, 12), #first layer
    nn.ReLU(),        
    nn.Linear(12, 8),#second layer
    nn.ReLU(),
    nn.Linear(8, 1), #output layer
    nn.Sigmoid()
)

print(model)

class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

model = PimaClassifier()
#print(model)

n_input_layer = 8
n_layer1 = 12
n_layer2 = 8
n_output_layer = 1

loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.SGD(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10
loss_every_epoch = []

for epoch in range(n_epochs):
    loss_in_current_epoch = 9999.0
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        ybatch = y[i:i+batch_size]
        y_pred = model(Xbatch)
        optimizer.zero_grad()
        loss = loss_fn(y_pred, ybatch)
        if(loss < loss_in_current_epoch):
            loss_in_current_epoch = loss.item()

        loss.backward()
        optimizer.step()
    loss_every_epoch.append(loss_in_current_epoch)
    print(f'Finished epoch {epoch}, latest loss {loss}')

xs = range(0, len(loss_every_epoch))

plt.plot(np.array(xs), np.array(loss_every_epoch))
plt.show()
# Make sure to close the plt object once done
plt.close()

with open('NeuralNetwork.csv','w') as f:
        writer = csv.writer(f)

        for key in loss_every_epoch:
            writer.writerow([key])

# compute accuracy (no_grad is optional)
X_test =  dataset[615:768,0:8]
Y_test =  dataset[615:768, 8]
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)

with torch.no_grad():

    y_pred = model(X)
    accuracy = (y_pred.round() == y).float().mean()
    print(f"Accuracy {accuracy}")

    y_pred_test = model(X_test)
    y_pred = model(X)

accuracy = (y_pred_test.round() == Y_test).float().mean()
print(f"Accuracy on test set {accuracy}")

accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy on training set {accuracy}")

# make class predictions with the model
predictions = (model(X) > 0.5).int()

for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

