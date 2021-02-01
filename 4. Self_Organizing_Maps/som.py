# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')

# X is all columns except last column, from customer ID to attribute 14
X = dataset.iloc[:, :-1].values

# y is whether the customer was approved, the last column
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X) # all values in X are normalized into (0, 1)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
# x = 10, y = 10 means we only need a 10 by 10 grid
# input_len is the number of features, here we have ID and 14 attributes, totally 15
# sigma is the radius of different neighbourhoods in the grid, set to default
# learning_rate is the alpha in ML.

som.random_weights_init(X) # randomize the initial weights
som.train_random(data = X, num_iteration = 100)



# this is code to define the number of cells to handle and retrieve the right values of these cells from the very brightest downwards.
# code from Udemy discussion:
# https://www.udemy.com/course/deeplearning/learn/lecture/8374816#questions/7422118
# get frauds automatically based on color
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





# Visualizing the results
# we'll get the Mean Interneuron Distance (MID) for winning nodes
# this is the mean of all nodes around the winning nodes 
# the higher MID, the more outlying it is
from pylab import bone, pcolor, colorbar, plot, show

# this will lay out a sketch for plot
bone()

# this will return the matrix of all distances for all winning nodes
pcolor(som.distance_map().T) # .T is taking the transpose of the MID matrix

# gives legend for the colors to tell which is high/low
# high MID's potentially are outliers and could be fraud
colorbar() 

markers = ['o', 's'] # o is circle, s is square

colors = ['r', 'g'] # r is red, g is green

for i, x in enumerate(X): 
    # i is all indexes of X, x is all vectors of customers at different iterations
    w = som.winner(x) # this gets the winning node of customer X
    plot(w[0] + 0.5,
         w[1] + 0.5, # those two lines puts the marker at the center of the graph square
         markers[y[i]], # so if y[i] is 0 then marker is 'o', if y[i] is 1 then marker is 's'
         markeredgecolor = colors[y[i]], 
         markerfacecolor = 'None', # no color inside marker, so no overlapping colors
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X) # returns the underlying customer data for all graph squares

# frauds = np.concatenate((mappings[(0,3)], mappings[(8,7)]), axis = 0)
# frauds = np.concatenate((mappings[(8,7)]), axis = 0)
# axis = 0 is concatenating vertically, which requres same column number

frauds = getColorList(som.distance_map(), mappings, 2)
frauds = sc.inverse_transform(frauds)