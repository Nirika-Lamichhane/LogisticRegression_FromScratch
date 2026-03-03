# src/main.py
import numpy as np
from logistic_regression import logisticregression
from data_utils import generate_3d_Datas
from pca import center_data, covariance_matrix, eiggen_decompostition, final_projection
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


X, y = generate_3d_Datas()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], X[y == 0][:, 2]) # this is of class 0
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], X[y == 1][:, 2]) # this is of class 1
plt.show()
# [:, 0] means we are taking all the rows and only the first column, [:, 1] means we are taking all the rows and only the second column and [:, 2] means we are taking all the rows and only the third column. This is how we are plotting the 3D scatter plot for both classes.

X_centered, mean = center_data(X)

cov_matrix = covariance_matrix(X_centered)

eigenvalues, eigenvectors = eiggen_decompostition(cov_matrix)

X_pca = final_projection(X_centered, eigenvectors, n_components=2)


from logistic_regression import logisticregression

# Train model
model = logisticregression(learning_rate=0.1, n_iters=2000)
model.fit(X_pca, y)

# Predict
predictions = model.prediction(X_pca)

# Accuracy
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)