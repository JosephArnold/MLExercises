from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import sys


# read csv input file
input_data = pd.read_csv("MLTraining/Iris-data.csv", sep=",",header=None)

neighbors = NearestNeighbors(n_neighbors=10)
neighbors_fit = neighbors.fit(input_data)
distances, indices = neighbors_fit.kneighbors(input_data)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)

plt.show()

