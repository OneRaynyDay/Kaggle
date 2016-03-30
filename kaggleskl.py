import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import interp

from smote import *

from sklearn import decomposition 
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import grid_search
from sklearn.metrics import roc_curve, auc, f1_score

train_set = genfromtxt('train.csv', dtype=float, delimiter = ',' );
#get rid of names
train_set = np.delete(train_set, (0),  axis=0)

#get labels and get rid of target
target = train_set[:, -1]
ones = train_set[train_set[:, -1] ==1]

train_set = np.delete(train_set, (370), axis=1)
ones = np.delete(ones, (370), axis=1)

#get rid of ids
train_set = np.delete(train_set, (0), axis=1) 

# clean zero columns
#train_set = train_set[:, ~np.all(train_set==0, axis=0)]

#SMOTE PROCESSING
train_set = np.concatenate((train_set, SMOTE(ones, 1000, 4)))

print("Splitting data set up")

print("Preprocessing data")
#TODO preprocessing has numerical error
X_st = preprocessing.scale(train_set);
#grid search for alpha
#parameters = {'alpha' : [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'l1_ratio' : [0.1 , 0.25, 0.5, 0.75, 0.9], 'n_iter' : [10, 20, 30, 50], 'loss' : ['hinge', 'squared_hinge', 'log', 'huber', 'modified_huber']}

#classifier with ROC analysis
cv = cross_validation.StratifiedKFold(target, n_folds=2)
clf = linear_model.SGDClassifier(alpha=0.0001, n_iter=30, l1_ratio=0.75, loss='hinge', penalty='elasticnet', fit_intercept=True, shuffle=True, verbose=0, epsilon=0.1,n_jobs=-1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight='balanced', warm_start=False, average=False)
#clf = grid_search.GridSearchCV(sgd, parameters, scoring='f1')

mean_tpr = 0.0
mean_fpr = np.linspace(0,1,100)
all_tpr=[]

for i, (train, test) in enumerate(cv):
    probas_ = clf.fit(X_st[train], target[train]).predict(X_st[test])
    print("Getting "+str(i)+ " CV train set score: " + str(clf.score(X_st[train], target[train])))
    print("Getting "+str(i)+ " CV test set score: " + str(clf.score(X_st[test], target[test])))
    #compute ROC curve
    fpr, tpr, threshoulds = roc_curve(target[test], probas_)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

#print(clf.best_params_)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
