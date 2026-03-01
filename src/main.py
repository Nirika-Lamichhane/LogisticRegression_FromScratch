# src/main.py
import numpy as np
from data_utils import generate_3d_Datas
from pca import center_data, covariance_matrix, eiggen_decompostition, final_projection

X,y = generate_3d_Datas()

X_centered, mean = center_data(X)

cov_matrix = covariance_matrix(X_centered)

eigenvalues, eigenvectors = eiggen_decompostition(cov_matrix)

X_pca = final_projection(X_centered, eigenvectors, n_components=2)


print("original shape of dataset:",X.shape)
print(" centered shape :", X_centered.shape)
print("covariance matrix shape:", cov_matrix.shape)
print("eigenvalues shape:", eigenvalues.shape)
print("eigenvectors shape:", eigenvectors.shape)
print(" top 2 eigenvectrs: \n ",eigenvectors[:,:2])
print("final projected shape:", X_pca.shape)
