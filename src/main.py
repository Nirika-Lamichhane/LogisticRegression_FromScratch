# src/main.py
import numpy as np
from data_utils import generate_3d_Datas
from pca import center_data

X,y = generate_3d_Datas()
print("Original Data Shape:", X.shape)
print("First 5 samples:\n", X[:5])
print("Labels:\n", y[:5])

# 2️⃣ Center the data
X_centered, mean = center_data(X)
print("\nCentered Data (first 5 samples):\n", X_centered[:5])
print("Feature means (should be ~0):", np.mean(X_centered, axis=0))