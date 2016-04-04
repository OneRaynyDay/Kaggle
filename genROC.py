import numpy as np
from numpy import genfromtxt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import interp
from sklearn.kernel_approximation import RBFSampler
from sklearn import decomposition, cross_validation, svm, preprocessing, grid_search
from sklearn.metrics import roc_curve, auc, f1_score

fpr, tpr= 0.2, 0.8
roc_auc = (tpr-fpr+1)/2
print("TPR: " + str(tpr) + " FPR: " + str(fpr) + " AUC: " + str(roc_auc))
plt.figure(figsize=(1,1))
plt.plot([0,0,1], [0,1,1], lw=1, label='Perfect ROC (area = %0.2f)' %  100)
plt.plot([0,fpr,1], [0,tpr,1], lw=1, label='Realistic ROC CV (area = %0.2f)' %  roc_auc)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck (area = %0.2f)' %.50)
plt.plot([0,tpr,1], [0,fpr,1], lw=1, label='Worse than luck (area = %0.2f)' %  roc_auc)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
