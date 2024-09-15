!pip install pygam pandas matplotlib

import pandas as pd
import numpy as np
from pygam import LinearGAM, s
import pickle


# load train data 
train_data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv')
test_data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')
train_data.head()

# Features: year, month, day, hour
X_train = train_data[['year', 'month', 'day', 'hour']]

# Target: trips
y_train = train_data['trips']

# Prepare test data in the same way
X_test = test_data[['year', 'month', 'day', 'hour']]

# Define fit the GAM model
model = LinearGAM(s(0) + s(1) + s(2) + s(3))
modelFit = model.fit(X_train, y_train)
print(modelFit.summary())

# Generate predictions using the fitted model
pred = modelFit.predict(X_test)
# Store predictions in the test data
test_data['predicted_trips'] = pred

#Save the model to a file
with open('modelFit.pkl', 'wb') as pkl_file:
    pickle.dump(modelFit, pkl_file)
