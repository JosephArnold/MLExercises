import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import csv


# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

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

n_epochs = 100
batch_size = 16
n_input_layer = 8
n_layer1 = 12
n_layer2 = 8
n_output_layer = 1
num_of_particles = 100
dimensions = 221

loss_fn = nn.BCELoss()  # binary cross entropy
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.SGD(model.parameters(), lr=0.001)

#To print the weights
#https://stackoverflow.com/questions/67722328/how-to-check-the-output-gradient-by-each-layer-in-pytorch-in-my-code


#initialise particles in the swarm with random positions
#Choose an arbitary number of particles 'N'= 10
#Each particle will have the following number of dimensions = (12 * 8) + (8 * 12) + (1 * 8) + 12 + 8 + 1 = 221

#initialize particle's positions and initial velocity 'vt' randomly
particle_positions = torch.rand(num_of_particles, dimensions) * 0.001
vt =  torch.rand(num_of_particles, dimensions) 
vt_1 =  torch.rand(num_of_particles, dimensions)
vt_next = torch.rand(num_of_particles, dimensions)

#initialize global and particle best positions
global_best_position = torch.zeros(1, dimensions)
particle_best_positions = torch.zeros(num_of_particles, dimensions)
global_best_error =  9999.0
particle_best_error = torch.full((num_of_particles, 1), 9999.0)

#initialize scalar values of inertia weight 'w', cognitive weight 'c1', global weight 'c2' and random factors 'r1' and 'r2'
w = 0.9
lamda = 0.8
weight_decr_step = (0.9 - 0.4) / ((len(X) / batch_size) * n_epochs)
c1 = 1.9
c2 = 1.845
loss_every_epoch = []

print(X.shape[0])
for epoch in range(n_epochs):
    best_in_current_epoch = 9999
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        ybatch = y[i:i+batch_size]

        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        #run forward prop for each particle
        for j in range(0, num_of_particles): 
            
            #update weights of particles
            #get position of particle(i) and store it in model weight and biases
            curr_weight = particle_positions[j]
            #print(curr_weight)
            base = 0
            dimensions_of_w1 = n_input_layer * n_layer1
            model[0].weight.data = curr_weight[base : base + dimensions_of_w1].reshape(n_layer1, n_input_layer)
            model[0].bias.data =  curr_weight[base + dimensions_of_w1 : base + dimensions_of_w1 + n_layer1].reshape(1, n_layer1)

            base += dimensions_of_w1 + n_layer1
            dimensions_of_w2 = n_layer2 * n_layer1
            model[2].weight.data = curr_weight[base : base + dimensions_of_w2].reshape(n_layer2, n_layer1)
            model[2].bias.data =  curr_weight[base + dimensions_of_w2 : base + dimensions_of_w2 + n_layer2].reshape(1, n_layer2)

            base += dimensions_of_w2 + n_layer2
            dimensions_of_w3 = n_output_layer * n_layer2
            model[4].weight.data = curr_weight[base : base + dimensions_of_w3].reshape(n_output_layer, n_layer2)
            model[4].bias.data =  curr_weight[base + dimensions_of_w3 : base + dimensions_of_w3 + n_output_layer].reshape(1, n_output_layer)

            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch) 
             
        #    Use PSO to optimize the weights here. Reference given below. 
        #    https://pyswarms.readthedocs.io/en/latest/examples/usecases/train_neural_network.html
        #    1. Update particle best position and global best position based on error
        #    2. Update velocity
        #       v(t+1) = (w * v(t)) + (c1 * r1 * (p(t) – x(t)) + (c2 * r2 * (g(t) – x(t))
        #    3. Update position
            if(loss < best_in_current_epoch):
                best_in_current_epoch = loss.item()

            if(loss < particle_best_error[j]):
                particle_best_error[j] = loss
                particle_best_positions[j] = curr_weight
            if(loss < global_best_error):
                global_best_error = loss
                global_best_position = curr_weight
            
            vt_next[j] = (1.0 - lamda) * (vt[j] + (c1 * r1 * (particle_best_positions[j] - curr_weight)) + (c2 * r2 * (global_best_position - curr_weight))) + lamda * vt_1[j]
           
            particle_positions[j] = particle_positions[j] + vt_next[j]      

            vt_1[j] = vt[j]
            vt[j] = vt_next[j]
            
        #print(model[0].weight.shape)
        #print(model[0].bias.shape)
        #print(model[2].weight.shape)
        #print(model[2].bias.shape)
        #print(model[4].weight.shape)
        #print(model[4].bias.shape)
        #print(f' latest loss {loss}')
        #exit()
    loss_every_epoch = loss_every_epoch + [best_in_current_epoch]
    print(f'Finished epoch {epoch}, latest loss {global_best_error}')

xs = range(0, len(loss_every_epoch))

plt.plot(np.array(xs), np.array(loss_every_epoch))
plt.show()
# Make sure to close the plt object once done
plt.close()

with open('PSOMomentum1.csv','w') as f:
        writer = csv.writer(f)

        for key in loss_every_epoch:
            writer.writerow([key])

X_test =  dataset[615:768,0:8]
Y_test =  dataset[615:768, 8]
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)

with torch.no_grad():

    curr_weight = global_best_position
    base = 0
    dimensions_of_w1 = n_input_layer * n_layer1
    model[0].weight.data = curr_weight[base : base + dimensions_of_w1].reshape(n_layer1, n_input_layer)
    model[0].bias.data =  curr_weight[base + dimensions_of_w1 : base + dimensions_of_w1 + n_layer1].reshape(1, n_layer1)

    base += dimensions_of_w1 + n_layer1
    dimensions_of_w2 = n_layer2 * n_layer1
    model[2].weight.data = curr_weight[base : base + dimensions_of_w2].reshape(n_layer2, n_layer1)
    model[2].bias.data =  curr_weight[base + dimensions_of_w2 : base + dimensions_of_w2 + n_layer2].reshape(1, n_layer2)

    base += dimensions_of_w2 + n_layer2
    dimensions_of_w3 = n_output_layer * n_layer2
    model[4].weight.data = curr_weight[base : base + dimensions_of_w3].reshape(n_output_layer, n_layer2)
    model[4].bias.data =  curr_weight[base + dimensions_of_w3 : base + dimensions_of_w3 + n_output_layer].reshape(1, n_output_layer)

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

