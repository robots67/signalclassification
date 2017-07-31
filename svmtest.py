import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
np.set_printoptions(suppress=True, threshold=np.inf)

#load model created in svmtrain.py
classifier = joblib.load('svmmodel.pkl')

#load testing dataset
test_dataframe = pd.read_csv('MilkWaterTest.csv')

#Parse labels and features from test set
test_labels = test_dataframe.signal_type
labels = list(set(test_labels))
test_labels = np.array([labels.index(x) for x in test_labels])

test_features = test_dataframe.iloc[3:4,1:]
#test_features = np.array(test_features)

#Model testing
if classifier.predict(test_features) == 0:
    print("Milk")
else:
    print("Water")
"""print("Accuracy: " + str(classifier.score(test_features,test_labels)*100))"""