# src/main.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from data_utils import generate_3d_Datas
from pca import center_data, covariance_matrix, eiggen_decompostition, final_projection

X,y = generate_3d_Datas()

X_centered, mean = center_data(X)

cov_matrix = covariance_matrix(X_centered)

eigenvalues, eigenvectors = eiggen_decompostition(cov_matrix)

X_pca = final_projection(X_centered, eigenvectors, n_components=2)


from logistic_regression import logisticregression

# Train model
model = logisticregression(learning_rate=0.1, n_iters=2000)
model.fit(X_pca, y)

# Predict
predictions = model.presiction(X_pca)

# Accuracy
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)