# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data

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

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1 
# unrated needs to be number other than 0 or 1, so we assign -1 to it

training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
# ratings with 1 or 2 are movies not liked, >=3 stars means like the movie

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        # nv = number of visible nodes, nh = number of hidden nodes
        
        # initialize weights to size of nh by nv matrix tensor, ~N(0,1)
        self.W = torch.randn(nh, nv) 
        
        # bias of hidden nodes
        # i.e. probability of hidden nodes given visible nodes, P(hidden|visible)
        # 1 bias for each hidden node, and we have nh hidden nodes, so it's 1 by nh tensor
        # 1st dimension of 1 corresponds to the batch, 2nd dimension corresponds to bias
        self.a = torch.randn(1, nh)  
        
        # bias of visible nodes
        # i.e. probability of visible nodes given hidden nodes, P(visible|hidden)
        self.b = torch.randn(1, nv)
        
    def sample_h(self, x):
        # used to sample hidden nodes according to P(hidden|visible)
        # this is essentially a sigmoid function, 
        # we'll approximate the log likelihood through Gibbs sampling
        
        # for example we have 100 hidden nodes in our RBM,
        # the function will sample the activations of these hidden nodes 
        # according to a certain probability that we'll compute in the same function
        # for each nodes the probability is P(h|v)
        
        # sigmoid function applied to wx + a, 
        # wx is the weights vector times matrix x of visible neurons
        # a is the bias of hidden nodes
        # we use torch.mm to take the product of two torch tensors
        # so wx is actually product of x and self.W (transposed)
        wx = torch.mm(x, self.W.t())
        
        # inside the activation function is wx + bias
        # each input vector will not be treated individually, but inside batches
        # we need to make sure that the bias is applied to each line of the mini batch
        # so we need to expand the bias as wx
        activation = wx + self.a.expand_as(wx)
        # this activation function represents a probability 
        # Pr that the hidden node will be activated according to value of visible node
        
        # now we calculate P(h|v) 
        p_h_given_v = torch.sigmoid(activation)
        
        # final step, return the probability and sample of hidden neurons
        # we are making a Bernoulli RBM because we return only binary outcome of 1 or 0 (like or dislike)
        # for example if P(h|v) is 0.7, and we take a random number between 0 and 1
        # then if the random number < 0.7 we activate the neuron, vice versa
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        # this is very similar to sample_h
        # this is predicting visible nodes based on hidden nodes
        
        # here when calculating wy we do NOT transpose self.W
        # self.W is the weight matrix of P(v|h), so no need for transposing
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    
    def train(self, v0, vk, ph0, phk):
        # v0: input vector of ratings of all movies by one user (we'll loop all users)
        # vk: visible nodes obtained after k samplings (k iterations or k contrastive divergence)
        # ph0: vector of probabilities that, at the first iteration the hidden nodes = 1 given the values of v0, i.e. Pr(H_i = 1|v_0)
        # phk: vector of probabilities after k samplings given vector of visible nodes vk, i.e. Pr(H_i = 1|v_k)
        
        # we'll update (in order) our tensor weights, bias b (visible), bias a (hidden)
        # following are using steps given by AltRBM-proof.pdf, page 28
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
        
# define parameters for the RBM
nv = len(training_set[0]) # 1682 movies means 1682 visible nodes, so nv = 1682
nh = 100 # number of features that we want to detect, could be tuned
batch_size = 100 # update weights after 100 samples
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10 # number of epoch
for epoch in range(1, nb_epoch + 1):
    train_loss = 0 # using simple difference in abs values to measure loss 
    s = 0. # make s a float, s serves as a counter. We'll need to normalize the train loss by dividing the count of the number of users that rated at least 1 movie
    for id_user in range(0, nb_users - batch_size, batch_size):
        # looping over all users, we take users at batch size
        
        # vk would be the output of the sampling, but first is an input
        vk = training_set[id_user:id_user+batch_size] # from id_user to next 100
        v0 = training_set[id_user:id_user+batch_size] 
        ph0,_ = rbm.sample_h(v0) # initial probabilities, Pr(hidden = 1|real ratings)
        # ph0,_ the "_" is used so that it only returns the first element of sample_h
        
        for k in range(10):
            # this is the loop for k-step contrastive divergence
            # we'll sample over the hidden nodes and visible nodes
            _,hk = rbm.sample_h(vk) # update hidden nodes based on visible nodes
            _,vk = rbm.sample_v(hk) # update visible nodes based on hidden nodes
            vk[v0<0] = v0[v0<0] # we don't want to train using ratings of -1 (unrated) so we freeze them as -1 by assigning the value for each loop
            
        phk,_ = rbm.sample_h(vk) # similar as code line 184
        rbm.train(v0, vk, ph0, phk) 
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0])) # this is using ABS
        
        # we could also use RMSE for train_loss:
        # train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2))
        
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM using the test set
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1] # v is the input used to activate the hidden neurons, so we'll use the inputs of the training set
    vt = test_set[id_user:id_user+1] # vt is the output
    if len(vt[vt>=0]) > 0:
        # get ratings that are existent (0 or 1) on the target to make predictions
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        
        # again we could use RMSE for test_loss:
        # test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2))
        
        s += 1.
print('test loss: '+str(test_loss/s))
