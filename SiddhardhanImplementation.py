# Siddhardhans Implementation: Uses Logistic Regression Model to Train Dataset

import torch
# importing the dependencies
import numpy as np
# used to create panda data frame 
import pandas as pd
# has dataset ? Imports dataset we found on kaggle to numpy arrays 
import sklearn.datasets
# Used in order to split dataset into training data and testing data
from sklearn.model_selection import train_test_split
# Using Logistic Regression Algorithm to the Training Set
from sklearn.linear_model import LogisticRegression
# Used to see how many correct predictions our model is making
from sklearn.metrics import accuracy_score



### DATA COLLECTION & PROCESSING ###

# loading the data from sklearn (The breastcancer dataset we use is general)
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset)

# loading to panda data frame - for convience it will be easier for us to analyze the data 
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

# prints the first five rows of the dataframe
print(data_frame.head())

# adding the 'target' column to the data frame
# can name as diagnosis or whatever you like
# target is the array that contains the 0 and 1 values 
data_frame['label'] = breast_cancer_dataset.target

# print last 5 rows of the dataframe
print(data_frame.tail())

# number of rows and columns in the dataset
print(data_frame.shape)

# getting some information about the data
data_frame.info()

# checking for missing values
print(data_frame.isnull().sum())

# statistical measures about the data
print(data_frame.describe())

# checking the distribution of Target Varibale
# how many malihnat and how many benign 
print(data_frame['label'].value_counts())

# all the values for 0 will be taken 
# mean value will be determiend for all the columns 
print(data_frame.groupby('label').mean())


### SEPARATING THE FEATURES AND TARGET ###

# drop label column becuase we need all the feautres seperately 
# axis value would be 0 if we were dropping a row
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

print("This is X: \n")
print(X)
print("This is Y: \n")
print(Y)



### SPLITTING THE DATA INTO TRAINING DATA & TESTING DATA ###
#     train_test_split: Split arrays or matrices into random train and test subsets.

# we want 20% of data to be our test data 
# 80% is training data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)


### MODEL TRAINING ###
# ------- We can make our own LR model -------- #
model = LogisticRegression()

# training the Logistic Regression model using Training data
# goes through data repeatedly and tries to find a fit for paricular dataset
model.fit(X_train, Y_train)


### MODEL EVALUATION ###
# for classifaction problmes we mostly use a accuracy score
# accuracy on training data


# create an array called x train predicition 
# predict used to predict the label given the x values 
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data = ', training_data_accuracy)


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on test data = ', test_data_accuracy)

### BUILDING A PREDICTIVE SYSTEM ###

# picked a specific labels feautres ( this one is malignant)
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')

