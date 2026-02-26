# I am generating 3D synthetic data for binary classification to know how the data has to be ceneterd and generated for the binary classification
'''
The goal of this code is to :
generate the dataset and the label for each point in the cluster so that model can learn,
if a point is in this cluster then it is class 0 and if not here class 1.
'''

import numpy as np

def generate_3d_Datas(
        n_samples : int = 500,     # each cluster has 500 so total samples is 1000
        mean_offset : float =2.0,    # this means how far the two clusters are located
        random_seed : int = 42
):
    
    np.random.seed(random_seed)

    cluster_1 = np.random.randn(n_samples, 3)    # randn() this generates the numbers from a standard normal distribution i.e. mean = 0, spread around 0 and bell curve distribution.
    cluster_2 = np.random.randn(n_samples,3)

    # we have to shift the datasets into two different clusters which is done by adding the clusters by changing the sign as

    cluster_1 += np.array([mean_offset , mean_offset, mean_offset])
    cluster_2 += np.array([-mean_offset, -mean_offset, -mean_offset])

    # now we have to combine the both cluster into one dataset as:
    x = np.vstack((cluster_1, cluster_2))      # this is vertical stack as we keep each row on top of another for the total samples

    # we have to create the label for the clusters as binary classification is supervised
    y = np.hstack((
        np.zeros(n_samples),
        np.ones(n_samples)
    ))
    return x,y

# hstack and vstack requires the single argument as a tuple or list of arrays to be stacked, so we have to put the arrays in the tuple or list.