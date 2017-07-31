# signalclassification
Machine learning algorithms used for classifying electrical signals. Created for Keio-NUS CUTE Center.


This repository contains different Python programs that can be used to classify electrical signals.

Signals must:
1. Be stored as a .csv file
2. Have first box of the first row must contain the label "signal_type"
3. Have the first column act as labels for the type of signal
4. Be of uniform length for each training/testing session
5. Be stored in the same location as the machine learning classifier

To use a program, simply change the filename in pd.read_csv([FILENAME]) to
the file you are training/testing on. The name must be inputed as a string with file extension.
(i.e. "signaltest.csv" with quotes)

The different files:

svmclassifier.py :
The base svm classifier that trains AND tests datasets. The training and testing datasets must be
unique and seperate. Running the program will both train and test, requiring you to train every time.
Will print the prediction results as well as a score represented through the % correct.

svmtrain.py & svmtest.py :
Similar to the base svm classifier but with model persistence. To use, first run svmtrain.py with the training
dataset to create the model (which must be saved as a .pkl file). Then, load the test dataset and the .pkl model
file into svmtest.py. This method does not require training every time so long as you intend to use the same training
data set between tests.

datasetgenerator.py :
Use for generating different types of datasets. Each type of dataset is described in the comments within each method.
To use, simply replace which ever method is at the very bottom of the program with the dataset method you wish to use.
For exmple, if you want to generate the filtered datasets, at the very bottom, replace whatever methods are there with
genFilteredSet() and then run. Similarly, you can import datasetgenerator.py into other programs and generate using the
defined methods. Some datasets may take longer to generate than others (particularly the filtered sets).

