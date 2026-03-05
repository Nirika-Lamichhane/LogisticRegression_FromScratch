# here 3d representation is being written

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_data(X, y):

    # plotting the raw data before transformation
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], X[y == 0][:, 2], color='blue', label='Class 0', alpha=0.5)
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], X[y == 1][:, 2], color='red', label='Class 1', alpha=0.5)

    '''
    As the projection is in 3D so the method scatter has (x,y,z) coordinate
    and here we have X[y==0][:,0] which is the x coordinate of class 0 
    i.e. all rows of 1st columns and likewise we have y and z 
    
    '''
    
