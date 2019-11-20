# Self Organizing Map



# Fraud Detection ->
# Dataset contains information from customers applying for a credit card
# By the end, we need to give list of customers that cheated 
# The frauds are the outlying neurons, they are further from the neurons that "follow the rules"



# Import the Libraries ->
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 

# Import the Dataset ->
dataset = pd.read_csv('Credit_Card_Applications.csv') # UCI Machine Learning Repository
X = dataset.iloc[:, :-1].values #All columns except the last
y = dataset.iloc[:, -1].values #Only the last column


# Feature Scaling ->
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) 
    # Similar to the RNN use normalization. Get all features between 0 and 1.
    #Range needs to be between 0 and 1, which is normalization.
X = sc.fit_transform(X) 
    #Then fit the SC Object to X so that SC gets all the info from X (min, max, etc.)
    # All the info it needs to apply normalization to X


# Training the SOM ->
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5) 
    # x and y are the dimensions of the grid. Build a 10x10 grid.
    # Input Len = # of features we have in our dataset. Dataset X. X contains 14 attributes plus the Customer ID.
    # Sigma = Radius of the different neighborhoods in the grid.
    # Learning Rate = hyperparameter that decides how much the weights are updated during each iteration. 
            # Higher = The faster there will be convergence. Less Creative
            # Lower = More time to build. More Creative.
    # Dont need a random seed.
som.random_weights_init(X)
    # Initialize the Weights
som.train_random(data = X, num_iteration = 100)
    # Method to train the SOM - Train_Random
    # After this step, the SOM is trained


# Visualizing the Results ->
from pylab import bone, pcolor, colorbar, plot, show
bone()    
pcolor(som.distance_map().T) #Will return the matrix for all the nodes. Transposes Matrix.
colorbar()    
markers = ['o','s'] # Vector of 2 elements - Circle and Square 
colors = ['r','g'] # Vector of 2 elements - Red and Green
for i, x in enumerate(X): # i = diff values of all indexes of customer DB. x = diff vectors (rows) of customers.
    w = som.winner(x) # First get winning node for first customer. Returns a winning node for a specific customer.
        # On the winning node, plot the marker
    plot(w[0] + 0.5, # Coordinates of the winning node. X Coordinate. The 0.5 puts it at the center of the square.
         w[1] + 0.5, # Y Coordinate. The 0.5 puts it at the center of the square.
         markers[y[i]], # Y Vector. I is the index of the customer. if the customer doesnt get approved, y[i] = 0. 
         markeredgecolor = colors[y[i]], # Red if rejected, Green if approved.
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)    
show()    

    
# Finding the frauds ->
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
    # Use a dictionary that will contain all the mappings     
    # Use coordinates of outlying winning node. mappings[(8,1)]
    # Second outlying winning node is in cells 6, 8. Outlying winning node = Green Circle in White Square.
    # Use Concatenate to combine 2 mappings of these 2 customers. 
    # axis = 0 -> The axis were the arrays will be joined.
frauds = sc.inverse_transform(frauds)
    # Inverse the scaling - use the inverse_transform method
    # Reverses the scaling