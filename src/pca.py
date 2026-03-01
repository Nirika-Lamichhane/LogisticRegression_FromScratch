import numpy as np 

def center_data (X : np.ndarray):

    # we have to compute the mean of each feature i.e. the columns
    mean = np.mean(X, axis=0) # we used axis = 0 cause we have to go downwards to find the total values in the single feature

    # now centering the data
    X_centered = X - mean
    return X_centered, mean


def covariance_matrix (X_centered : np.ndarray):
    return np.cov(X_centered, rowvar=False) 

def eiggen_decompostition(cov_matrix):

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues [idx]
    eigenvectors = eigenvectors[:,idx]  # this means keep all the rows as it is but columns should be paired with the idx i.e. eigenvalues
    return eigenvalues, eigenvectors

def final_projection (X_centered, eigenvectors, n_components=2):
    top_eigenvectors =eigenvectors[:,:n_components]
    X_pca =X_centered @ top_eigenvectors
    return X_pca


'''
X :np.ndarray is just for the documentation purpose and to make the IDE autocomplete 
It is python's type hint syntax as:
def function_name (variable_name : expected_type):


np.cov computes the covariance of the all centered data and features, it gives us a matrix
eigenvalues is 1D and eigen vector is 2D

eigenvectors[:, [1, 2, 0]]

# give me column 1 first  → [0.8, 0.1, 0.6]
# give me column 2 second → [0.5, 0.3, 0.7]
# give me column 0 third  → [0.2, 0.9, 0.3]

# Result:
[[ 0.8,  0.5,  0.2],
 [ 0.1,  0.3,  0.9],
 [ 0.6,  0.7,  0.3]]
```

Now everything is correctly paired:
- Column 0 `[0.8, 0.1, 0.6]` paired with eigenvalue `8.7` ✅
- Column 1 `[0.5, 0.3, 0.7]` paired with eigenvalue `3.2` ✅
- Column 2 `[0.2, 0.9, 0.3]` paired with eigenvalue `0.5` ✅

---

## Full Workflow Summary
```
covariance matrix (3×3)
        ↓
np.linalg.eigh()
        ↓
eigenvalues [0.5, 8.7, 3.2]   ← unsorted, wrong order
eigenvectors (3×3)             ← columns misaligned
        ↓
np.argsort()[::-1]
        ↓
idx = [1, 2, 0]               ← sorting recipe
        ↓
eigenvalues[idx]               ← eigenvalues sorted ✅
eigenvectors[:, idx]           ← columns reordered ✅
        ↓
eigenvalues = [8.7, 3.2, 0.5]
eigenvectors columns reordered and correctly paired

'''
