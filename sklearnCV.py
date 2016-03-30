import numpy as np
from numpy import genfromtxt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import interp

from smote import * 

from sklearn.kernel_approximation import RBFSampler
from sklearn import decomposition, cross_validation, svm, preprocessing, grid_search
from sklearn.metrics import roc_curve, auc, f1_score


print("LOADING DATA")
data_set = genfromtxt('train.csv', dtype=float, delimiter = ',' );

#get rid of names, ids, and zero columns
data_set = np.delete(data_set, (0),  axis=0)
data_set = np.delete(data_set, (0), axis=1) 
data_set = data_set[:, ~np.all(data_set==0, axis=0)]

train_set, cv_set = cross_validation.train_test_split(data_set, test_size=0.4, random_state=0)

#get target, onez and zeros from dataset
y_train= train_set[:, -1]
ones = train_set[train_set[:, -1] ==1]
zeros = train_set[train_set[:, -1] ==0]

#clean target from dataset and get training set
X_train = np.delete(train_set, (-1), axis=1)
ones = np.delete(ones, (-1), axis=1)
zeros = np.delete(zeros, (-1), axis=1)

#get cv se
y_cv= cv_set[:, -1]
X_cv= np.delete(cv_set, (-1), axis=1)

print("ALL DATA LOADED")
print(X_train.shape)
print(y_train.shape)
print(X_cv.shape)
print(y_cv.shape)

print("PREPROCESSING DATA")
#SMOTE PROCESSING
#PCA ANALYSIS
scaler = preprocessing.StandardScaler(copy=False).fit(X_train)
scaler.transform(ones)
scaler.transform(zeros)
scaler.transform(X_train)
scaler.transform(X_cv)

X_pca = decomposition.PCA(n_components=2).fit_transform(X_train)
y_orig = y_train;


#scale_factor = 0
#X_train = np.concatenate((X_train, SMOTE(ones, scale_factor*100, 5)))
#y_train = np.concatenate((y_train, np.ones(ones.shape[0]*scale_factor)))

print(X_train.shape)
print(y_train.shape)

X_pca_smote = decomposition.PCA(n_components=2).fit_transform(X_train)

for X_transformed, y, title in [(X_pca, y_orig, "PCA"), (X_pca_smote,y_train, "SMOTE PCA")]:
    plt.figure(figsize=(10,10))
    for c, i , l in zip("rb", [0,1], ["Satisfied", "Unsatisfied"]):
        plt.scatter(X_transformed[y==i, 0], X_transformed[y==i,1],c=c,label= l)

    plt.title(title+" of datset (Size: %d)" % y.size)
    plt.legend(loc="best")
    plt.autoscale(enable=True)

plt.show()

print("FINISHED PREPROCESSING DATA")
#grid search for alpha
#parameters = {'alpha' : [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'l1_ratio' : [0.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'class_weight' : ['balanced', None]}

#RBF KERNEL
#rbf_feature = RBFSampler(n_components=400, gamma=1, random_state=1)
#rbf_feature.fit(X_train);
#X_train = rbf_feature.transform(X_train)
#X_cv = rbf_feature.transform(X_cv)

print(X_train.shape)
print(X_cv.shape)

#SVM classifier with ROC analysis
#clf = linear_model.SGDClassifier(alpha=0.01, l1_ratio=0.1, loss='hinge', n_iter=15, penalty='elasticnet', fit_intercept=True, shuffle=True, verbose=0, epsilon=0.1,n_jobs=-1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False)
#clf = grid_search.GridSearchCV(sgd, parameters, scoring='roc_auc')

#PASSIVE AGRESSIVE CLASSIFIER
#clf = linear_model.PassiveAggressiveClassifier(fit_intercept=False, n_iter=20, loss='hinge_squared')
#parameters = { 'gamma' : np.exp2(np.arange(-15,3,3))}
clf = svm.SVC(C=8192, gamma=0.00192)
#clf = grid_search.GridSearchCV(sgd, parameters, scoring='roc_auc', n_jobs = -1)

print("TRAINING CLASSIFIER")
clf.fit(X_train, y_train)
probas_ = clf.predict(X_cv)
print("FINISEHED TRAINING CLASSIFIER")

print("Getting training set score: " + str(clf.score(X_train, y_train)))
print("Getting CV set score: " + str(clf.score(X_cv, y_cv)))

#print("BEST PARANS: " + str(clf.best_params_))
#compute ROC curve
fpr, tpr, threshoulds = roc_curve(y_cv, probas_)
roc_auc = auc(fpr, tpr)
print("TPR: " + str(tpr) + " FPR: " + str(fpr) + " AUC: " + str(roc_auc))
plt.figure(figsize=(1,1))
plt.plot(fpr, tpr, lw=1, label='ROC CV Set (area = %0.2f)' %  roc_auc)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
