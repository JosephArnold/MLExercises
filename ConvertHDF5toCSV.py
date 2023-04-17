#!/usr/bin/env python3
import h5py
import pandas as pd
import numpy as np

with h5py.File('/home/joseph/bremen_small.h5','r') as hf:
    print('List of arrays in this file: \n', hf.keys())
np.savetxt('out.csv', h5py.File('/home/joseph/bremen_small.h5')['DBSCAN'], '%g', ',')


#read h5 file
#dataset = h5py.File('/home/joseph/flies_D_F-h5.hdf', 'r')

#print the first unkown key in the h5 file
#print(dataset.keys())

#print the keys inside the first unkown key
#df = dataset['data']
#print(df) #prints sub list keys such as axis0 and axis1

#save the h5 file to csv using the first key df
#with pd.HDFStore('/home/joseph/flies_D_F-h5.hdf', 'r') as d:
#    df = d.get('data')
#    df.to_csv('metr-la.csv')
#print the attributes of keys such as axis0 inside the first unkown key
#print("axis0 data: {}".format(df['axis0']))
#print("axis0 data attributes: {}".format(list(df['axis0'].attrs)))
