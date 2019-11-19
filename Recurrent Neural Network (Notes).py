# Recurrent Neural Network (Notes)



# Predict Google Stock Price
# Train LSTM model on 5 years of Google Stock Data
# Predict January 2017 - Then test against Jan 2017 Test Data
# Predict Upward or Downward Trends - Not the Stock Price

# Recurrent NN's don't just learn from the weights.
# They also learn from past data.



# Model Overview ->

# Data Preprocessing ->
    # Import Libraries
    # Import Training Set
    # Feature Scaling
    # Data Structure with Timesteps and Output
    # Reshaping

# Building RNN ->
    # Import Keras
    # Sequential
    # Add First LSTM Layer
    # Add Droput 1
    # Add Second LSTM Layer
    # Add Droput 2
    # Add Third LSTM Layer
    # Add Dropout 3
    # Add Fourth LSTM Layer
    # Add Droput 4
    # Add the Output Layer
    # Compile the Optimizer and Loss Function (Adam and Mean Squared Error)
    # Fit the RNN to the Training Set (X_train and y_train)    
    
# Predictions and Visualizations ->
    # Loan Test Set - Build Dataframe - Get Real Stock Price of 2017
    # Get Predicted Stock Price of 2017
    # Visualize the Results
    


# Part 1 - Data Preprocessing ->

# Import Libraries ->
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Training Set ->
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv') 
training_set = dataset_train.iloc[:, 1:2].values

    # Feature Scaling ->
    # Standardization vs. Normalization - Use Normalization with RNN's
    # Use feature scaling to optimize the training process

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) # sc for Scale. Create sc object using MinMaxScaler
training_set_scaled = sc.fit_transform(training_set)    # To normalize it, fit the sc object to the training object
                                                        # Fit Means it grabs the Min and Max of Stock Price to apply to Normalize Function
                                                        # training_set_scaled dataset is now a normalized dataset
                                                            
# Creating a data structure with 60 timesteps and 1 output ->
X_train = [] # Input of Neural Network
y_train = [] # Output of Neural Network
for i in range(60, 1258):   # 60 is for the previous 60 stock prices; 1,258 is the upper bound - look at # of rows in dataset
    X_train.append(training_set_scaled[i-60:i, 0]) # append 60 previous stock prices before the i'th financial day. i minus 60 to i. Specify column 0 at end.
    y_train.append(training_set_scaled[i, 0]) # Line,Index.   i = Line, 0 = Index. 
X_train, y_train = np.array(X_train), np.array(y_train) # Converts X_train and y_train to numpy arrays

    # Important - Wrong number of timestamps can lead to overfitting or non-sense predictions

    # Look at the X_Train dataset -> Each new day shows yesterdays score before it, and the one before that.
    # Look for diagonal pattern in dataset.
    
    # 1 timestamp leads to overfitting
    # 20 isn't enough either
    # 60 worked best - 60 previous past financial days - 20 finance days in each month -so- 60 time stamps = last 3 months
    
    # x_train - past 60 stock prices
    # y_train - next stock price

# Reshaping ->
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # "Reshape" the X_train variable:
    
    # First X_train - what we want to reshape
    # Next X_train.shape[0] - This will give you the total # of rows from the X_train dataset
    # Then X_train.shape[1] - This will give you the total # of columns from the X_train dataset
    # Finally - 1 - Number of Indicators/Predictors - 1 Open stock price
    
    # Visualize this in 3D - 3D Tensor with shape - Keras documentation helps with this.
    
    # This is for the number of predictors that we want. These predictors are indicators. So far we have 1. The open stock price. 
    # We're using the 60 prior to try and predict that open stock price.
    
    # X_train now has 3 Dimensions ->
        # Click the Dataset in the Variable Explorer
        # Adjust the Axis: at the bottom left between 0, 1, 2 to see the different dimensions
        
        # 3 Dimensions = observations, timesteps, and indicators



# Part 2 - Building the RNN ->
        
    # Stacked LSTM Model
    # Quote - "All models are wrong, but some are useful"

# Importing the Keras libraries and packages ->
from keras.models import Sequential   # Creates neural network object representing sequence of layers
from keras.layers import Dense        # Adds Output Layer
from keras.layers import LSTM         # Adds LSTM layers
from keras.layers import Dropout      # Adds Dropout Regularization

# Initialising the RNN
regressor = Sequential()
    # Initializes the Regressor
    # Use sequential class from Keras
    # "regressor" is an object of the sequential class, which is a sequence of layers
        # Regression - Predicting a continuous value
        # Classification - Predicting a category or class

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    # 3 Arguments
    # 1 - # of Units - LSTM Cells - In this Layer
    # 2 - Return sequences - Set to True - Building stacked LSTM, when adding another, the return sequence needs to be on True
    # 3 - Input Shape - Shape of the input containing x_train - 3 Dimensions corresponding to observations, timesteps, and indicators

    # Units = 50 - 50 neurons will give model with high dimensionality - 3-5 would be too small and wouldnt capture up and down trends
    # Return Sequences - True - Since were building stacked RNN and were adding another LSTM layer so this needs to be true
    # Input Shape - (X_train.shape[1], 1) - last part for first LSTM layer

regressor.add(Dropout(0.2))
    # Dropout Regularization to prevent overfitting - dont want when predicting stock prices
    
    # Add the dropout rate - 0.2
    # The # of neurons to ignore to complete the regularization
    # Drop 20% of them
    # 20% of the neurons of the LSTM layer will be ignored during the training - 10 neurons will be dropped out

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

    # Dont need to specify the Input_Shape in the new layers
    # Only need to specify the units and return_sequences
    
    # Dropout still set at 20%

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
    # In the last LSTM layer, don't need to set return_sequences to True

# Adding the output layer
regressor.add(Dense(units = 1))
    # Adding the output layer
    # Theres 1 output layer, so set units = 1 using dense function
    
    # Which class should we use?
    # Use dense class for a full connection

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # Compile - Method of sequential class - include 2 arguments
    # Optimizer and Loss Function
    # Optimizer = Adam = Performs relevant updates of the weights - Stochastic Optimization
    # Loss Function = Mean Squared Error - Since its a regression problem, use Mean Squared Error

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    # Intelligent, trained NN to predict upward and downward trends of Googles Stock Price
    # Use the fit functino
    # X_train - Independent Variables - Forward Propragated to Output
    # y_train - Dependent Variable
    # Epochs - How many iterations to train the neural network. Here 100 times. Good amount for 5 years of stock price data.
    # Batch Size - NN is trained on batches of iterations. Every 32 stock prices, model is trained. 



# Part 3 - Making the predictions and visualising the results ->

# Getting the real stock price of 2017 ->
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
    # Convert Dataset into Numpy Data Frame


# Getting the predicted stock price of 2017 ->
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    # Concatinate the Training dataset and Test dataset - the way we do this matters
    # Dont concatinate to the real_stock_price dataset. Need to concatinate the training dataset to the test dataset
    # RNN was trained on scaled values
    # So for consistency, we need to use scaled inputs to get predictions. Normalization with SC Object.
        
    # dataset_train - identify the 'Open' column
    # dataset_test - identify the 'Open' column
    # axis = 0 -> Concatinate along the vertical axis which = 0 (horizontal = 1)
        
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    # Get the inputs - At each day, we need to get the stock prices of the previous 60 days

inputs = inputs.reshape(-1,1)
    # Uses inputs function above    
    # Use reshape to get right numpy shape - Need to better understand the -1 and 1

inputs = sc.transform(inputs)
    # Transform into 3D format
    # Scale the values - satisifies 3rd key point - only scale the input, not the test values

X_test = []
    # Input for test set

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
    # Range Lower Bound and Upper Bound
    # Lower Bound stays at 60
    # Upper Bound changes from 1258 to 80 (60 previous inputs + 20 new inputs)
    # Then append X_test using Inputs
    
X_test = np.array(X_test)
    # X_test into a numpy array

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # Reshape into 3D Structure

predicted_stock_price = regressor.predict(X_test)
    # Predict the Stock Price using Predict Regressor

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    # Inverse the scaling of our predictions
    # Use Inverse Transform Method



# Visualising the results ->
    
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



# RMSE for Model Evaluation ->
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))





















