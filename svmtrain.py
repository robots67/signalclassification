import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
np.set_printoptions(suppress=True, threshold=np.inf)

#load training dataset
data = pd.read_csv('MilkWaterTraining.csv')

#Parse labels and features from training set
train_labels = data.signal_type
labels = list(set(train_labels))
train_labels = np.array([labels.index(x) for x in train_labels])

train_features = data.iloc[:,1:]
train_features = np.array(train_features)

#Begin training and testing
classifier = svm.SVC(kernel='linear')
classifier.fit(train_features, train_labels)

#Export training model for persistence
joblib.dump(classifier, 'DemoModel.pkl')
#Model can now be loaded in testing program using the .pkl file
