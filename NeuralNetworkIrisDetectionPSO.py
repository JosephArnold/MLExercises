import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import csv

plt.style.use('ggplot')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

# Scale data to have mean 0 and variance 1
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)

n_input_layer = X_train.shape[1]
n_layer1 = 50
n_layer2 = 50
n_output_layer = 3
num_of_particles = 100
dimensions = n_input_layer * n_layer1 * n_layer2 * n_output_layer + (n_input_layer + n_layer1 + n_layer1 + n_output_layer)
particle_positions = torch.rand(num_of_particles, dimensions) * 0.01
vt =  torch.rand(num_of_particles, dimensions)
c1 = 0.4
c2 = 0.3

#initialize global and particle best positions
global_best_position = torch.zeros(1, dimensions)
particle_best_positions = torch.zeros(num_of_particles, dimensions)
global_best_error =  9999.0
particle_best_error = torch.full((num_of_particles, 1), 9999.0)

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

model     = Model(X_train.shape[1])
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.CrossEntropyLoss()

import tqdm

EPOCHS  = 100
X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test  = Variable(torch.from_numpy(X_test)).float()
y_test  = Variable(torch.from_numpy(y_test)).long()

loss_list     = np.zeros((EPOCHS,))
accuracy_list = np.zeros((EPOCHS,))

for epoch in tqdm.trange(EPOCHS):
    
    best_in_current_epoch = 9999.0
    for j in range(0, num_of_particles):
            #update weights of particles
            #get position of particle(i) and store it in model weight and biases
            curr_weight = particle_positions[j]
            #print(curr_weight)
            base = 0
            dimensions_of_w1 = n_input_layer * n_layer1
            model.layer1.weight.data = curr_weight[base : base + dimensions_of_w1].reshape(n_layer1, n_input_layer)
            model.layer1.bias.data =  curr_weight[base + dimensions_of_w1 : base + dimensions_of_w1 + n_layer1].reshape(1, n_layer1)

            base += dimensions_of_w1 + n_layer1
            dimensions_of_w2 = n_layer2 * n_layer1
            model.layer2.weight.data = curr_weight[base : base + dimensions_of_w2].reshape(n_layer2, n_layer1)
            model.layer2.bias.data =  curr_weight[base + dimensions_of_w2 : base + dimensions_of_w2 + n_layer2].reshape(1, n_layer2)

            base += dimensions_of_w2 + n_layer2
            dimensions_of_w3 = n_output_layer * n_layer2
            model.layer3.weight.data = curr_weight[base : base + dimensions_of_w3].reshape(n_output_layer, n_layer2)
            model.layer3.bias.data =  curr_weight[base + dimensions_of_w3 : base + dimensions_of_w3 + n_output_layer].reshape(1, n_output_layer)


            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            

            if(loss < best_in_current_epoch):
                best_in_current_epoch = loss.item()
            if(loss <= particle_best_error[j]):
                particle_best_error[j] = loss.item()
                particle_best_positions[j] = curr_weight
            if(loss <= global_best_error):
                global_best_error = loss.item()
                global_best_position = curr_weight

    loss_list[epoch] = best_in_current_epoch

    #particle position updated after current epoch
    for j in range(0, num_of_particles):
        w =  random.uniform(0.8, 0.8)
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        curr_weight = particle_positions[j]
        vt[j] = (w * vt[j] + ((c1 * r1 * (particle_best_positions[j] - curr_weight)) + (c2 * r2 * (global_best_position - curr_weight))))
        particle_positions[j] = particle_positions[j] + vt[j]

    with torch.no_grad():
        y_pred = model(X_test)
        correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list[epoch] = correct.mean()

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

ax1.plot(accuracy_list)
ax1.set_ylabel("validation accuracy")
ax2.plot(loss_list)
ax2.set_ylabel("validation loss")
ax2.set_xlabel("epochs")
plt.show()

print(f'Best accuracy {max(accuracy_list)}')
