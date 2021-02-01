# Mega Case Study - Make a Hybrid Deep Learning Model



# Part 1 - Identify the Frauds with the Self-Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

from collections import defaultdict
def getColorList(gridMap, mapping, index=1):
    
    
    #    Info:       
    #        extention to get Grid-Coords of populated Cells,\n
    #        depending on the Grid-Color and user defined Grid-Count.\n
    #        after Pcolor(som.distance_map().T) is called, an Instance of the Grid-System is going to be initialized.\n
    #        After this Initialisation you can access the Grid values, which by the way also includes the coords and color-code.
    #                
    #    Parameter:  
    #        gridMap as Instance of Grid-System: by calling 'som.distance_map()'.\n
    #        mapping as defaultdict: dictionary of the returning grid-mapping-call\n
    #        index as int() / default=1: number of cells to be evaluated
        
    
    
    # variable declaration
    colorGrid = {}
    concLoop = 1
    concate = []
    myFraud = None
    sortColors = defaultdict(tuple)
    
    # loop to extract color-value and corresponding grid-cell
    # color-value as float64, from 0. > 1.
    # where 0 relates to black and 1 to white. 
    # depending on the color-value you can work on specific cells
    for i in range(len(gridMap)):
        for j in range(len(gridMap)):
            for key in mapping.keys():
                # check if cell is populated, if true write cell to dict
                colTuple = (i,j)
                if key == colTuple:
                    colorGrid.update({gridMap[i][j]: [i, j]})
    
    # colorGrid is ordered 'ASC' on color-value
    # needs to be reversed to have the brightest cells first
    # this loop takes care
    for k in sorted(colorGrid.keys(), reverse=True):
        sortColors[k] = (colorGrid[k][0], colorGrid[k][1])
 
    # this loop feeds the Fraud by checking whether only the brightest cell shall be evaluated 
    # or a specified number of cells more.
    for v in sortColors.values():
        # breaks when the number of cells has been reached
        if concLoop > index: break
        
        # cells > 1 every additional cell will be mapped and stored in a List
        if index > 1:
            #print(v) # uncommend to get the handled cells printed to the console
            concate.append(mapping[v])
            concLoop += 1
        else:
            # default value or 1 cell was passed, map the brightest cell and break the loop
            myFraud = mapping[v]
            break
    
    # check if more than 1 cells was passed in.
    # List-Comprehension adds all List-entries and concatenates it
    if index > 1: myFraud = np.concatenate(([x for x in concate]), axis = 0)
    
    return myFraud 



# Finding the frauds
mappings = som.win_map(X)
frauds = getColorList(som.distance_map(), mappings, 2)
frauds = sc.inverse_transform(frauds)



# Part 2 - Going from Unsupervised to Supervised Deep Learning

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values # remove customer ID column, leaving only attributes

# Creating the dependent variable
is_fraud = np.zeros(len(dataset)) # len(dataset) returns the row number of dataset
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds: 
        # dataset.iloc[i,0] would return the customer ID 
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# The following codes are from ANN for supervised deep learning 
# see ann.py, part 2, make the ANN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)


# Predicting the probabilities of frauds
y_pred = classifier.predict(customers)

y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
# we need to make a horizontal concatenation so axis = 1

y_pred = y_pred[y_pred[:, 1].argsort()] # sort by column 2