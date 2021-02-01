# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
 
labelencoder_x1 = LabelEncoder()
labelencoder_x2 = LabelEncoder()
  
# Gender label encode
X[:, 2] = labelencoder_x2.fit_transform(X[:, 2])

# Country label encode, this maps the three countries to 0, 1, 2
X[:, 1] = labelencoder_x1.fit_transform(X[:, 1])
 
# Problem with mapping countries to 0, 1, 2 is that it introduced higher or lower, but in fact Germany is not necessarily higher or lower than France, so we need to separate this into columns of only 1 or 0 to indicate if the customer belongs to a certain country. This is called country column transform
ct = ColumnTransformer([('Geography', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# This country column transform changes the country column into three columns of 0 or 1, i.e. France = 0 or 1, Spain = 0 or 1, Germany = 0 or 1
# Problem is, we only need two columns (if a country is not France or Spain then it must be Germany), so we discard the first column
# This is called "Dummy variable trap"
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# The scaling is calculated by z = (x - u)/s, where u is the sample mean and s is the sample standard deviation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





# Part 2 - Make the ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform", input_dim=11))
# parameter explanation in Dense(): 
# input_dim: is the number of input dimension; 
# units: is the number of nodes in the hidden layer, usually the average of input and output dimensions, (11 + 1) / 2 = 6 (the choose of number of hidden layer is subjective), an old syntax calls 'units' 'output_dim'
# Kernel_initializer: The neural network needs to start with some weights and then iteratively update them to better values. The term kernel_initializer is a fancy term for which statistical distribution or function to use for initialising the weights. In case of statistical distribution, the library will generate numbers from that statistical distribution and use as starting weights. Here we use uniform distribution for initialize.
### Different kernel initializer ###
# https://keras.io/initializers/#randomnormal

### Different activation functions ###
# https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
# softmax is equivalent to sigmoid, but for >1 nodes; they are both logistic, see the link above

# Adding the second hidden layer
# We don't need to specify the input_dim because it's already stated in the first layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding the output layer
# only 1 node, so units = 1
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# by doing the above steps we have a ANN with structure of 11-6-6-1

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# loss function is the sum of all (y - y_hat)^2
# for the loss function, if there are >2 outcomes, e.g. 3 categories, then the loss would be 'categorical_crossentropy'
# here we only predict whether the customer leaves the bank or not, it's just 1 or 0, so we use binary_crossentropy
# metrics = ['accuracy'] is the method we choose to evaluate the model, typically we just choose accuracy so that when weights updated after each observation or each batch of many observations, the algorithm uses the accuracy criterion to improve the metrics
# metrics is a list, so we need the []

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# The batch_size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters. A sample is one single row of data
# The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.
# so here we are working at 10 rows at a time, and will train for 100 times





# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# convert the y_pred from 0 to 1, to only 0 or 1 in order to use the confusion matrix, so we are predicting if a client is staying or leaving the bank, so we changed the probability from 0 to 1 to a binary of true (leaving) or false (staying)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
# we will be able to see if our training is accurate. (0,0) and (1,1) are correct predictions (predict leaving actually leaving, or predicting staying actually staying)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Given new data, predict the new observation
    # Predict if the customer will leave the bank:
    # Geography: France
    # Credit Score: 600
    # Gender: Male
    # Age: 40
    # Tenure: 3
    # Balance: 60000
    # Number of Products: 2
    # Has Credit Card: Yes
    # Is Active Member: Yes
    # Estimated Salary: 50000


# turn the info into a row, use [[]] of the array() as below, which means a list of one list
# if we need to test multiple new customers, the form would be like [[], [], ..., []]
# first need to check matrix X to get the order correct
new_prediction = classifier.predict(
    # we need to apply the scaling as in line 46, X_test = sc.transform(X_test)
    sc.transform(
        np.array(
            # to see which column France belongs to, compare X and dataset
            # same logic for male/female
            # we could see the order is:
            # is_Germany, is_Spain, Credit Score, is_male, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary
            [[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
            )
        )
    )
new_prediction = (new_prediction > 0.5)





# Part 4 - Evaluating, Improving & Tuning the ANN
# this is improving the ANN by decreasing the variance of the accuracy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    # this builds the structure of the previous ANN into a defined function
    classifier = Sequential()
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform", input_dim=11))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# insert this line when running keras and TF with sklearn under Windows 
# the famous "mian" function lol
if __name__ == "__main__":
    classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
    # return the 10 accuracies of the 10 test folds that occur in K-fold cross validation
    # estimator: the object to use to fit the data
    # cv: number of train test folds when applying K-fold validation, usually 10 is used to produce 10 accuracies
    # n_jobs: # of CPUs used, -1 means all CPUs, could run parallel trainings
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, n_jobs = -1)
    mean = accuracies.mean()
    variance = accuracies.std()
    print("\nfinished")
    print("Mean: ",mean)
    print("Variance: ",variance)
 
# cleanup for TF
# import gc; gc.collect()






# Dropout regularization, some of the neurons got randomly disabled during iteration
# can be to one or several layers
# from keras.layers import Dropout
# Adding the input layer and the first hidden layer with dropout
# classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform", input_dim=11))
# classifer.add(Dropout(p = 0.1)) # it means that 10% of this layer will be randomly dropped out, p ranges from 0 to 1
# Adding the second hidden layer with dropout
# classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
# classifer.add(Dropout(p = 0.1))





# Tuning the ANN
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    # this builds the structure of the previous ANN into a defined function
    classifier = Sequential()
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform", input_dim=11))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    # !!! notice here, optimizer = optimizer
    # this is for the parameters input for other methods
    # notice the differences in lines 203 and 209, compared with lines 155 and 161
    return classifier

if __name__ == "__main__":
    classifier = KerasClassifier(build_fn = build_classifier)
    # because we are tuning and dropping out neurons, so it's different than previous code:
    # classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 10)
    # we don't need the statement for batch_size and epochs
    parameters = {'batch_size': [25, 32],
                  'epochs': [100, 500],
                  'optimizer': ['adam', 'rmsprop']}
    # here we are testing other batch_size (testing 25 and 32) and other epochs (testing 100 and 500), comparing to previous batch_size 10 and epochs 100
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10)
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    # we'll be able to compare results from different parameters
    # we are training 8 models (2 batch_size * 2 epochs * 2 optimizer)
