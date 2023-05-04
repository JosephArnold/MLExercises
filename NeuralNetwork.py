import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
#load only from column 0 to column 8
X = dataset[:,0:8]

#load column 8 which is the output vector
y = dataset[:,8]

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
print(model)

loss_fn = nn.BCELoss()  # binary cross entropy
#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.001)

n_epochs = 1
batch_size = 1

#To print the weights
#https://stackoverflow.com/questions/67722328/how-to-check-the-output-gradient-by-each-layer-in-pytorch-in-my-code
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

    print(f'Finished epoch {epoch}, latest loss {loss}')


# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
