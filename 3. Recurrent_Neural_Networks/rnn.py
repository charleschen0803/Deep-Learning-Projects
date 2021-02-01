# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values 
# we want to create a numpy array of 1 column, in python a:b is [a,b) in math
# so we are actually pulling all the "Open" column in dataset_train

# Feature Scaling
# in RNN we usually use "Normalization" instead of "Standardization"
# Normalization scales all data to ~[0,1]
# Normalization: (x - min)/(max - min)
# Standardization: (x - mean)/StDev
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# it means we are using 60 financial dates for all T for predicting the next financial date
# i.e. use data from [T-60, T) to predict T, in Python it's [i-60:i]
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# info on Keras about the input shape:
# https://keras.io/layers/recurrent/
# 3D tensor with shape (batch_size, timesteps, input_dim).
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_train.shape[0] is the batch_size, i.e. row number
# X_train.shape[1] is the timesteps, here is equal to column number
# 1 means we are only using 1 input dimension (Google stock price) for prediction
# after this, X_train would be a 3-D data structure


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential # create NN object
from keras.layers import Dense # output layer
from keras.layers import LSTM # LSTM layers
from keras.layers import Dropout # dropout regularisation

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# (1) units is the # of cells we want to have in the LSTM layers, we choose 50 to have high dimensionality which leads to better results
# (2) return_sequences: set to True if will be adding another LSTM layer after this one
# (3) input_shape only contains the last two dimensions - time steps and indicators (same as line 41)


# dropout regularisation is used for avoiding overfitting
regressor.add(Dropout(0.2))
# this drops out 20% of the neurons of the LSTM layer randomly, which is 50 * 20% = 10 neurons

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
# after this fourth and final LSTM layer, we are not adding more LSTM layers, so do not add the statement of return_sequences = True (it's default as False)
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# adam is always a safe optimizer choice
# we could also use RMSprop
# see details in: https://keras.io/optimizers/

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# epochs: # of times for the entire forward propagation & backward propagation

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
'''
we'll need data from both training set and test set, so we need to concatenate the two
we'll have some 60 previous days data from the training set for each day of 2017 Jan
RNN was trained on the scaled values of the train set, so need to use the scaled input to get the predictions.
'''

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# axis of 0 and 1 is whether we are concatenating along columns or rows
# vertical concatenation is axis = 0

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# at each financial date of Jan 2017 we need to get the 60 previous stock prices, which is the stock prices from the 3 past months (1 month has 20 financial days)
# e.g. to predict Jan 3, 2017 (this is the first financial date of 2017), we'll need to use data 60 days ago
# len(dataset_total) - len(dataset_test) is right at Jan 3, 2017
# [len(dataset_total) - len(dataset_test) - 60:] is 60 financial days before Jan 3, 2017 up until the last financial date, which results in 80 numbers

inputs = inputs.reshape(-1,1)
# this is used for reshaping the input list from one dimensional to 3 dimensional
# previously it was like [779, 780, ..., 797]
# after reshaping the inputs list becomes [[779], [780], ..., [797]]
# reshape(-1,1) means to shape it so that there is 1 column number, and unknown row number (-1 means unknown and Python would figure it out)
# for example if there is a list [1,2,3,...,12]
# reshape(-1,4) means to reshape so that there are 4 columns with unknown rows, and Python would figure it out to be 3 rows
# reshaping result would be [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
# see following for more details of why using reshape(-1,1)
# https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape

inputs = sc.transform(inputs)
# only scale the inputs but not the test values
# transform the data to 0 ~ 1

X_test = []
for i in range(60, 80):
    # appending the Jan 2017 data for X_test, with only 1 column
    # there will be 20 rows and each row contains 60 data
    X_test.append(inputs[i-60:i, 0])

# use numpy to make it into array
# will get a 20x60 array
X_test = np.array(X_test)

# transfer to 3D format, see line 41
# changes the size from (20,60) to (20,60,1)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# use the trained regressor to predict X_test, dimension is (20,1), value range from 0 to 1
predicted_stock_price = regressor.predict(X_test)

# inverse transform so that the value is scaled from 0~1 to normal price values
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Evaluating RNN using RMSE (Root Mean Squared Error)
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

### how to improving RNN 

# 1. Getting more training data: 
#     we trained our model on the past 5 years of the Google Stock Price but it would be even better to train it on the past 10 years.
#     
# 2. Increasing the number of timesteps: 
#     the model remembered the stock prices from the 60 previous financial days to predict the stock price of the next day. That’s because we chose a number of 60 timesteps (3 months). You could try to increase the number of timesteps, by choosing for example 120 timesteps (6 months).
#     
# 3. Adding some other indicators: 
#     if you have the financial instinct that the stock price of some other companies might be correlated to the one of Google, you could add this other stock price as a new indicator in the training data.
#     
# 4. Adding more LSTM layers: 
#     we built a RNN with four LSTM layers but you could try with even more.
#     
# 5. Adding more neurones in the LSTM layers: 
#     we highlighted the fact that we needed a high number of neurones in the LSTM layers to respond better to the complexity of the problem and we chose to include 50 neurones in each of our 4 LSTM layers. You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.


### how to tune the RNN

# Parameter Tuning for Regression is the same as Parameter Tuning for Classification which you learned in Part 1 - Artificial Neural Networks, the only difference is that you have to replace:
# 
# scoring = 'accuracy'  
# 
# by:
# 
# scoring = 'neg_mean_squared_error' 
# 
# in the GridSearchCV class parameters.


