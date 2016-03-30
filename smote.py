"""SMOTE implementation written by 
Jesse Cai 
can be used for both over and under sampling
"""
import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors

def SMOTE(T, N, k):
    #T is the number of minority class samples, N is the percentage smote, and k is the nearest neighbors
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        n_samples, n_features = T.shape
        #valuable numbers

        if N < 100:
            #TODO random sampling to get the difference
            pass
        elif N%100 == 0:
            N = (int)(N/100) # multiplication factor
        else: 
            raise ValueError("WRONG VALUE")

        #set the return matrix ( synthetic)
        S = np.zeros((N*n_samples, n_features))

        #fit for nearest neibotrs
        nbrs = NearestNeighbors(n_neighbors=k).fit(T)
        newindex = 0; 

        for i in range(n_samples):
            indicies =  nbrs.kneighbors(T[i, :], return_distance=False)

            for j  in range(N):
                nn = np.random.choice(indicies[0])
                while nn == i:
                    nn = np.random.choice(indicies[0])
                dif = T[nn, :] - T[i, :]# get a random sample of knn
                gap = np.random.rand()
                S[newindex, :] = T[i,:] + gap*dif[:]
                newindex+=1;
        return S
