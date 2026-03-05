# src/main.py
import numpy as np
from data_utils import generate_3d_Datas
from pca import center_data, covariance_matrix, eigen_decomposition, final_projection  
from logistic_regression import logisticregression 
from sklearn.model_selection import train_test_split
from visualization import plot_3d_data, plot_pca, plot_3d_vs_2d, plot_decision_boundary, plot_weight_projection, plot_margin

# data generation
X, y = generate_3d_Datas(mean_offset=0.8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
plot_3d_data(X, y) 

# centering using train data only
X_train_centered, mean = center_data(X_train)

# PCA on training data only
cov_matrix = covariance_matrix(X_train_centered)
eigenvalues, eigenvectors = eigen_decomposition(cov_matrix) 

plot_pca(X_train, y_train, eigenvectors, eigenvalues, mean)

X_train_pca = final_projection(X_train_centered, eigenvectors, n_components=2)  

plot_3d_vs_2d(X_train, y_train, X_train_pca, y_train)

# apply SAME transformation to test using train's mean and eigenvectors
X_test_centered = X_test - mean 
X_test_pca = final_projection(X_test_centered, eigenvectors, n_components=2)  

# train model on train set only
model = logisticregression(learning_rate=0.1, n_iters=2000)
model.fit(X_train_pca, y_train)  

plot_decision_boundary(X_train_pca, y_train, model)
plot_weight_projection(X_train_pca, y_train, model)
plot_margin(X_train_pca, y_train, model)

# evaluate on test set
predictions = model.prediction(X_test_pca)  
accuracy = np.mean(predictions == y_test)  
print("Test Accuracy:", accuracy)

'''
accuracy = np.mean(predictions == y_test)  
this single line of code performs 3 operations as:
1. it finds the predictions that is in 1 and 0 and the y test also in 1 and 0.
2. Then it compares and give true boolean where the numbers are same and false otherwise
3. Then the NumPy treats true as 1 and false as 0 and find the mean.
'''