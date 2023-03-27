from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt

# read csv input file
input_data = pd.read_csv("MLTraining/dataLAC.csv", sep=",",header=None)


#print(input_data)
dbscan = DBSCAN(eps = 0.013, min_samples = 11).fit(input_data) # fitting the model
labels = dbscan.labels_ # getting the labels

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

counts = np.bincount(labels[labels>=0])

print(counts)
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
# learning the clustering from the input date
#kmeans.fit(input_data.values)

#print(kmeans.labels_)


#input_data.to_csv("MLTraining/cluster_data_python100.csv")

#print(input_data)
# output the labels for the input data
#for index, value in enumerate(input_data):
#    print(str(index) + " " + value+"\n")# + " " + kmeans.labels_[index] + "\n")
