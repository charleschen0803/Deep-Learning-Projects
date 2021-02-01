# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
# default separator is ","
# in our dataset it's "::"

users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
# separator is a tab, so use the delimiter = '\t'
# the structure of the training_set is the same as ratings

training_set = np.array(training_set, dtype = 'int')
# here we convert the dataframe into arrays

# we then do the same for the test set
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
# We need this function for both training set and test set
def convert(data):
    # What will be returned is a list of lists
    # A list of lists of ratings
    new_data = []
    for id_users in range(1, nb_users + 1):
        
        id_movies = data[:,1][data[:,0] == id_users] 
        # this returns list of all movies of the user's ID in the loop
        
        id_ratings = data[:,2][data[:,0] == id_users]
        # this returns list of all ratings of the user's ID in the loop
        
        ratings = np.zeros(nb_movies)
        # creates a placeholder of an array of zeros
        
        ratings[id_movies - 1] = id_ratings
        # We need to - 1 for the id to match Python index
        # We assign each rated zeros in the ratings list with real ratings
        # For those movies not rated by user, it will remain 0
        
        new_data.append(list(ratings))
    return new_data


# Originally both training_set and test_set are arrays created in pandas 
# We used convert() to convert it to lists of lists for easier manipulation
# Both sets would be a list with nb_users lists in it, and within each lists in the list is a list of nb_movies ratings


training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ): # no additional parameters as we are considering inheritance
        super(SAE, self).__init__() 
        # super() is used to get the inherited methods from the nn.Module class
        
        self.fc1 = nn.Linear(nb_movies, 20)
        # fc1 means the first full connection related to the AutoEncoders object
        # 20 is the # of elements in the first encoded vector (hidden nodes)
        
        self.fc2 = nn.Linear(20, 10)
        # here we transferred the 20 nodes in first hidden nodes into 10 nodes in second hidden layer, this is the second encoding
        
        self.fc3 = nn.Linear(10, 20)
        # we are starting to decode to reconstruct the original vector
        
        self.fc4 = nn.Linear(20, nb_movies)
        # the final step of decoding
        
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        # this is used for the whole encoding & decoding process
        # we'll use the self.activation that's defined in def __init__ to do this
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x) # because it's the final step so we do not need activation
        return x
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
# lr stands for learning rate
# weight_decay is used to reduce the learning rate after every few epochs to regulate the convergence, so model could be improved even more

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # make s a float, s serves as a counter. We'll need to normalize the train loss by dividing the count of the number of users that rated at least 1 movie
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        # training_set[id_user] is a single vector of one dimension,
        # a network in PyTorch or Keras generally don't accept it
        # what is more accepted is a batch of input vectors, so we need to make it a batch
        # we use the Variable().unsqueeze(0) to do this
        # the .unsqueeze(0) is the index in the new dimension 
        
        target = input.clone() # this is just copying the input
        
        if torch.sum(target.data > 0) > 0:
            # this condition will help saving the memory by looking at 
            # users who at least rated 1 movie
            
            output = sae(input) # this uses the method of the class SAE that we defined
            
            target.require_grad = False
            # make sure that we don't compute the gradient with respect to the target
            # so it could save computations and optimize the codes
            
            output[target == 0] = 0
            # we don't want to consider movies that the user didn't rate for output vector
            # this also optimizes the codes
            # those values will not count in the computation of error
            
            loss = criterion(output, target) 
            # to calculate loss, we only need the true values and predicted values
            # output is prediction, target is truth
            
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            # float() converts the value to a float
            # torch.sum() is like sumif() in Excel, 
            # we need to make sure that the denominator is not null
            # the + 1e-10 is to make sure that denominator is not 0
            # the mean corrector is used for calculating the average of the error but by only considering the movies that were rated. The mean corrector is for adapting to this consideration, this will then be mathematically relevant to compute the mean of the errors 
            
            loss.backward() 
            # this indicates the direction of how we update the weights is backward
            # this is backpropagation
            
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1.
            
            optimizer.step()
            # apply the optimizer to update the weights, decides the amount of update
            
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))

# Testing the SAE
# codes are very similar to the training process, but do not need loop for epoches
test_loss = 0 
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))