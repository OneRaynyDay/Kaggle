import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import interp

from sklearn import decomposition 
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

train_set = genfromtxt('train.csv', dtype=float, delimiter = ',' );
#get rid of names
train_set = np.delete(train_set, (0),  axis=0)

#get labels and get rid of target
target = train_set[:, -1]
train_set = np.delete(train_set, (370), axis=1)

#get rid of ids
train_set = np.delete(train_set, (0), axis=1) 

# clean zero columns
train_set = train_set[:, ~np.all(train_set==0, axis=0)]
print("Splitting data set up")

print("Preprocessing data")
#TODO preprocessing has numerical error
X_st = preprocessing.scale(train_set);

#classifier with ROC analysis
cv = cross_validation.StratifiedKFold(target, n_folds=6)
clf = linear_model.SGDClassifier(alpha=0.0001, average=False, class_weight='balanced', epsilon=0.1, eta0=0.0, fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', loss='modified_huber', n_iter=100, n_jobs=-1, penalty='l2', power_t=0.5, random_state=None, shuffle=True, verbose=2, warm_start=False)

mean_tpr = 0.0
mean_fpr = np.linspace(0,1,100)
all_tpr=[]

for i, (train, test) in enumerate(cv):
    probas_ = clf.fit(X_st[train], target[train]).predict_proba(X_st[test])
    #compute ROC curve
    fpr, tpr, threshoulds = roc_curve(target[test], probas_[:,1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

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
