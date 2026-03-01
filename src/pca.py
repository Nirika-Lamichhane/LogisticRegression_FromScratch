import numpy as np 

def center_data (X : np.ndarray):

    # we have to compute the mean of each feature i.e. the columns
    mean = np.mean(X, axis=0) # we used axis = 0 cause we have to go downwards to find the total values in the single feature

    # now centering the data
    X_centered = X - mean
    return X_centered, mean


'''
X :np.ndarray is just for the documentation purpose and to make the IDE autocomplete 
It is python's type hint syntax as:
def function_name (variable_name : expected_type):

'''