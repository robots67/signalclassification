import pandas as pd
import numpy as np
from sklearn import svm
np.set_printoptions(suppress=True, threshold=np.inf)

def classify(train_file, test_file):

    if not isinstance(train_file,str) or not isinstance(test_file,str):
        raise Exception("File names must be entered as string")
    elif train_file[-4:] != '.csv' or test_file[-4:] != '.csv':
        raise Exception(
            "Files must be .csv files and be entered with a .csv extension")

    #load training dataset
    data = pd.read_csv(train_file)

    #Parse labels and features from training set
    train_labels = data.signal_type
    labels = list(set(train_labels))
    train_labels = np.array([labels.index(x) for x in train_labels])

    train_features = data.iloc[:,1:]
    train_features = np.array(train_features)

    #Begin training and testing
    classifier = svm.SVC(kernel='linear')
    classifier.fit(train_features, train_labels)

    #load testing dataset
    test_dataframe = pd.read_csv(test_file)

    #Parse labels and features from test set
    test_labels = test_dataframe.signal_type
    labels = list(set(test_labels))
    test_labels = np.array([labels.index(x) for x in test_labels])

    test_features = test_dataframe.iloc[:,1:]
    test_features = np.array(test_features)

    #Model testing
    print(classifier.predict(test_features))
    return classifier.score(test_features,test_labels)*100
