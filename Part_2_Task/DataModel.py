from sklearn.linear_model import LinearRegression
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.regularizers import l2,l1

from sklearn.cluster import KMeans  
from sklearn.mixture import GaussianMixture
import sys 
import os 
import random
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath("../Task_3_4_5"))
sys.path.append(os.path.abspath("../Task_1_2"))
from Data_Acquisition import DataAcquisition
from DataGenerator import DataGenerator


class DataModel :
    def __init__(self,n_p):
        self.class_data=DataGenerator(n_p)
        self.data=self.class_data.create_2d_list_linear()

    def linear_regressions(self):
        list_xy = np.array(self.class_data.list_xy)
        self.X = list_xy[:,0].reshape(-1,1) 
        self.y =  list_xy[:,1] # The list of value created 
        model = LinearRegression()
        model.fit(self.X,self.y)
        self.y_pred = model.predict(self.X)
        plt.figure(figsize=(8,6))
        plt.scatter(self.X, self.y, color='blue', label='real data')
        plt.plot(self.X, self.y_pred, color='red', linewidth=2, label='linear regression')
        plt.xlabel(' X coordonate of distribution')
        plt.ylabel(' Y coordonate of distribution')
        plt.title('Linear regression')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def train_validation_test(self,test_ratio=0.2,train_ratio=0.3,val_ratio=0.5,random_state=None):
        """ 
        Steps:
            1. Shuffle the data randomly (to avoid bias).
            2. Split into training, testing  and validating sets 
        Inputs : test_ratio -> set the size of test set 
                 train_ratio -> set the size of train set 
                 val_ratio_ratio -> set the size of validation set 

        """
        # Shufle the data
        if random_state is not None:
            random.seed(random_state)
        shuffled_data=list(self.X)
        random.shuffle(shuffled_data)

        # Calculate the values of each set 
        n = len(shuffled_data)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)
        train_set = shuffled_data[:train_end]
        val_set = shuffled_data[train_end:val_end]
        test_set = shuffled_data[val_end:]

        # Help with visualisation of data 
        print("Training Data:")
        for item in train_set:
            print(item)

        print("\nTesting Data:")
        for item in test_set:
            print(item)
        print("\Validating Data:")
        for item in val_set:
            print(item)
        print(f"\nLength of Training Data: {len(train_set)}")
        print(f"Length of Testing Data: {len(test_set)}")
        print(f"Length of Validation Data: {len(val_set)}")

    def mean_square_error(self):
        real_data = 2*self.X.ravel().astype(float) # Simulation the real linear coefficient of 2*x
        predicted_data = self.y.ravel().astype(float) # Data predicted with noise 
        print(real_data)
        print(predicted_data)
        error = np.square( predicted_data -real_data)
        mse = np.mean(error)
        print("mean square error", mse)

    def neural_network(self):
        self.Y=self.y.reshape(-1,1)
        model = Sequential(
            [
                Dense(8,activation='relu',input_shape=(1,),kernel_regularizer=l2(0.01)),
                Dense(64,activation='relu',input_shape=(1,),kernel_regularizer=l2(0.01)),
                Dense(1) # Output layer with 1 neuron for regression
            ]
        )

        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        model.fit(self.X,self.Y,epochs=20,verbose=1)
        y_pred_nn = model.predict(self.X)

    def k_mean(self,n_cluster):   
        """
        Function to calculate the Kmean Model
        K_mean is a Clustering algorithm based on the assumption that each cluster is represented by a spherical blob centered at its centroid
        """     
        model=KMeans(n_clusters=n_cluster)
        model.fit(self.X)
        self.y_kmeans=model.predict(self.X)
        plt.figure(figsize=(8,6))
        plt.scatter(self.X, self.Y, c=self.y_kmeans, label='K-Means Clustering')
        plt.title('K-Means Clustering')
    
    def gmm(self,n_component):
        """
        Function to calculate the Gaussian Mixture Model
        GMM is a Clustering algorithm based on the assumption that each cluster comes from a Gaussian distribution
        """
        model_gm=GaussianMixture(n_components=n_component)
        model_gm.fit(self.X) 
        y_gm=model_gm.predict(self.X)
        plt.figure(figsize=(8,6))
        plt.scatter(self.X, self.Y, c=y_gm, label='GMM Clustering')
        plt.title('GMM Clustering')
        


       

    
