from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from costcla.models import CostSensitiveRandomForestClassifier
from costcla.metrics import savings_score
import random
import numpy as np
import glob

train_yes_data = np.load("../../data/TNSCUI2020_train/classification_ds/vgg16_cropped512features_train_yes.npy")
train_yes_cnt = train_yes_data[:,0:-1].shape[0]
X_train_yes = train_yes_data[:, 0:-1]
X_train_yes = X_train_yes.astype(float)
y_train_yes = np.ones((train_yes_cnt,1), np.uint8)
#y_train_yes = y_train_yes.astype(str)
#print(y_train_yes)

train_no_data = np.load("../../data/TNSCUI2020_train/classification_ds/vgg16_cropped512features_train_no.npy")
train_no_cnt = train_no_data[:,0:-1].shape[0]
X_train_no = train_no_data[:, 0:-1]
X_train_no = X_train_no.astype(float)
y_train_no = np.zeros((train_no_cnt,1), np.uint8)
#y_train_no = y_train_no.astype(str)

valid_yes_data = np.load("../../data/TNSCUI2020_train/classification_ds/vgg16_cropped512features_valid_yes.npy")
valid_yes_cnt = valid_yes_data[:,0:-1].shape[0]
X_valid_yes = valid_yes_data[:, 0:-1]
X_valid_yes = X_valid_yes.astype(float)
y_valid_yes = np.ones((valid_yes_cnt,1), np.uint8)
#y_valid_yes = y_valid_yes.astype(str)

valid_no_data = np.load("../../data/TNSCUI2020_train/classification_ds/vgg16_cropped512features_valid_no.npy")
valid_no_cnt = valid_no_data[:,0:-1].shape[0]
X_valid_no = valid_no_data[:, 0:-1]
X_valid_no = X_valid_no.astype(float)
y_valid_no = np.zeros((valid_no_cnt, 1), np.uint8)
#y_valid_no = y_valid_no.astype(str)

#print(y_valid_no)
X_train = np.append(X_train_yes, X_train_no, axis=0)
y_train = np.append(y_train_yes, y_train_no, axis=0)

X_valid = np.append(X_valid_yes, X_valid_no, axis=0)
y_valid = np.append(y_valid_yes, y_valid_no, axis=0)

print(X_train.shape, y_train.shape)
temp = list(np.hstack((X_train, y_train)))
random.shuffle(temp)
#print(temp)
#print(temp.shape)
X_train = np.asarray(temp)[:,:514]
y_train = np.asarray(temp)[:, 514]
y_train = np.expand_dims(y_train, axis=1)
#y_valid = np.expand_dims(y_valid, axis=1)

#print(X_valid.shape)
#print(y_valid)

#exit(0)
y_train = np.squeeze(y_train)
y_valid = np.squeeze(y_valid)

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
#exit(0)
cost_mat_train=np.zeros((X_train.shape[0],4))
cost_mat_valid=np.zeros((X_valid.shape[0],4))

cost_mat_train[:,0]=3.28
cost_mat_train[:,1]=4.0

cost_mat_valid[:,0]=3.28
cost_mat_valid[:,1]=4.0

clf = CostSensitiveRandomForestClassifier(n_estimators=100)
#clf = DecisionTreeClassifier(random_state=0)
#clf = XGBClassifier(max_depth=1000, eta=0.1)
#clf = KMeans(n_clusters=2, random_state=0)
#clf = svm.SVC(kernel='poly', degree=3)
#clf = KNeighborsClassifier()
#clf = RandomForestClassifier(n_estimators=100, random_state=1)
m=clf.fit(X_train, y_train, cost_mat_train)
print(savings_score(y_train, m.predict(X_train), cost_mat_train))
print(savings_score(y_valid, m.predict(X_valid), cost_mat_valid))

#print(clf.score(X_valid, y_valid))
#print(cross_val_score(clf, X_train, y_train, cv=10))
#print(m.predict(X_valid))

