# Siddhardhans Implementation: Uses Logistic Regression Model to Train Dataset

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

# Data Collection & Processing 

# loading the data from sklearn (The breastcancer dataset we use is general)
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset)

