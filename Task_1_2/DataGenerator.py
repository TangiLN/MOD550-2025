import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataGenerator:
    def __init__(self,n_p=10):
        """Generate any data"""
        self.n_p=n_p

    def create_2d_list_random(self):
        """
        Create a 2D list of random number  with another method  
        Output: The random  list 
        """
        self.list_xy=[]
        for i in range(self.n_p):
            x=random.randint(0,self.n_p)
            y=random.randint(0,self.n_p)
            self.list_xy.append([x,y])
    
    def create_2d_list_linear(self):
        """
        Create a 2D list of random number  with linear distribution  
        Output: The random  list 
        """
        self.list_xy=[]
        for i in range(self.n_p):
            x=random.randint(0,self.n_p) # same as random.randint but with float numbers 
            noise= random.gauss(0,self.n_p/10 )#random floating point number with gaussian distribution.
            y=2*x+noise
            self.list_xy.append([x,y]) 

    def plotter(self):
        """Generate Plots"""
        x_vals = [point[0] for point in self.list_xy]
        y_vals = [point[1] for point in self.list_xy]
        plt.scatter(x_vals, y_vals)
        plt.show()

    def plotter_histogram(self):
        """Generate histograms plots"""
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
        bin_data = int(self.n_p/10)
        print("size of a bin : ",bin_data)
        x_vals = [point[0] for point in self.list_xy]
        y_vals = [point[1] for point in self.list_xy]

        axs[0].hist(x_vals,bins=bin_data,color="blue")
        axs[0].set_title("Histogram of X values ")

        axs[1].hist(y_vals,bins=bin_data,color="orange")
        axs[1].set_title("Histogram of Y values ")
        plt.tight_layout()
        plt.show()

    def plotter_heatmap(self):
        """Generate Heatmap plot"""
        bin_data=10    
        x_vals = [point[0] for point in self.list_xy]
        y_vals = [point[1] for point in self.list_xy]
        heatmap, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=bin_data)
        sns.heatmap(heatmap.T, annot=True, cmap="YlGnBu",cbar_kws={'label': 'Counts'})
        plt.xlabel("X bins")
        plt.ylabel("Y bins")
        plt.title("Heatmap plot")
        plt.show()

