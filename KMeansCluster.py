from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt

# read csv input file
input_data = pd.read_csv("MLTraining/knearest.csv", sep=",",header=None)


#print(input_data)
# initialize KMeans object specifying the number of desired clusters
#kmeans = KMeans(n_clusters=4)


# learning the clustering from the input date
#kmeans.fit(input_data.values)

#print(kmeans.labels_)

#input_data["labels"] = kmeans.labels_

#input_data.to_csv("MLTraining/cluster_data_python100.csv")

plt.plot(input_data)
plt.show()
#print(input_data)
# output the labels for the input data
#for index, value in enumerate(input_data):
#    print(str(index) + " " + value+"\n")# + " " + kmeans.labels_[index] + "\n")
