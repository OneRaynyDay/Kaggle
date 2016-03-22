
# coding: utf-8

# **Kaggle Competition - Santander**

# In[5]:

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict # for the data, we don't want to have to create a new array


# Reading in the data:

# In[10]:

raw_data = []
train_data = defaultdict(list)
test_data = defaultdict(list)

def read_file(name):
    '''
    Params: name of file
    returns: tuple [keys from the first line of csv file, rest of the lines as arr]
    '''
    data_keys = []
    raw_data = []
    # Reading in the keys of the value
    with open(name, 'r') as f:
        lines = f.readlines()
        for k in lines[0].split(","):
            data_keys.append(k)
        raw_data = lines[1:]
    
    return [data_keys, raw_data]

def ingest_data(data_keys, data_arr, result):
    '''
    Params: keys from first line, data from the rest of lines, result in the form of a dict(defaultdict allowed)
    returns: None
    '''
    # Taking in the data using the keys
    for row in data_arr:
        for key, dat in zip(data_keys, row):
            result[key].append(float(dat))

key, raw_train_data = read_file('train.csv')
key, raw_test_data = read_file('test.csv')

ingest_data(key, raw_train_data, train_data)
ingest_data(key, raw_test_data, test_data)
# Test to see the size of train_data
print(len(train_data))
print(len(test_data))


# Data preprocessing

# In[ ]:



