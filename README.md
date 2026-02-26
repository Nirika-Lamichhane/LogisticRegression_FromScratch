
# Geometric Machine Learning Pipeline

### PCA + Logistic Regression from Scratch (Vector & Matrix View)

---

##  Overview

This project builds a **Geometric Machine Learning pipeline** from scratch using only vector and matrix transformations.

Instead of treating ML as a black box, this project explains:

* What PCA *really* does geometrically
* What Logistic Regression *really* learns
* How projections define both dimensionality reduction and classification
* Why centering and covariance matter

The goal is to understand machine learning as **linear algebra + geometry**.

---

#  What We Are Building

We construct the following pipeline:

```
Raw High-Dimensional Data
        ↓
Center Data (Subtract Mean μ)
        ↓
PCA (Project onto Principal Components)
        ↓
Reduced 2D Representation
        ↓
Logistic Regression (Linear Classifier)
        ↓
Decision Boundary Visualization
        ↓
Projection Geometry Visualization
```

This shows how **projection** is the core operation behind both:

* PCA (unsupervised projection)
* Logistic Regression (supervised projection)

---

# Step-by-Step Explanation

---

## 1️⃣ Raw Data (3D or Higher)

We generate synthetic 3D vectors:

[
x_i = (x_{i1}, x_{i2}, x_{i3})
]

Each data point is a vector in ℝ³.

Why 3D?

Because it allows us to:

* Visualize dimensionality reduction clearly
* Show projection from 3D → 2D
* Build geometric intuition

---

## 2️⃣ Centering the Data

We compute the mean vector:

[
\mu = \frac{1}{n} \sum_{i=1}^{n} x_i
]

Then subtract it:

[
X_{centered} = X - \mu
]

### Why?
If we donot subtract the mean, **we measure the spread relative to origin, not relative to the true center which gives the wrong directions.**

#### Centering removes location information and preserves only shape information.

PCA requires data to be centered because:

* Variance is measured relative to the mean
* The covariance matrix assumes zero-mean data
* Eigenvectors represent true directions of spread only after centering

### Geometric Meaning

Centering shifts the entire data cloud so that:

[
\text{Mean} = 0
]

The cloud now sits around the origin.

---

## 3️⃣ PCA – Principal Component Analysis

PCA finds directions where the data varies the most.

Mathematically:

1. Compute covariance matrix:

[
\Sigma = \frac{1}{n} X_{centered}^T X_{centered}
]

2. Compute eigenvalues and eigenvectors:

[
\Sigma v = \lambda v
]

3. Sort eigenvectors by descending eigenvalues.

4. Project data:

[
Z = X_{centered} W
]

Where:

* (W) = matrix of top principal components
* (Z) = reduced data

---

### What PCA Is Geometrically

PCA is:

> Projection onto orthogonal directions of maximum variance.

It finds the best 2D plane that preserves maximum information from 3D.

We are not deleting information randomly.

We are keeping the directions where data spreads most.

---

## 4️⃣ Reduced 2D Data

Now each point becomes:

[
z_i = (z_{i1}, z_{i2})
]

We have compressed 3D → 2D.

This step:

* Reduces dimensionality
* Preserves maximum variance
* Makes visualization easier

---

## 5️⃣ Logistic Regression (From Scratch)

Now we classify the reduced data.

Model:

[
\hat{y} = \sigma(w^T z + b)
]

Where:

* (w) = weight vector
* (b) = bias
* (\sigma(x) = \frac{1}{1+e^{-x}})

---

### What Logistic Regression Is Geometrically

Logistic regression:

* Projects each point onto vector (w)
* Applies a sigmoid
* Learns a decision boundary:

[
w^T z + b = 0
]

This is a **line in 2D space**.

So again:

It is projection + thresholding.

---

#  The Core Idea: Projection Everywhere

| Algorithm           | What It Projects    | Why               |
| ------------------- | ------------------- | ----------------- |
| PCA                 | Onto principal axes | Preserve variance |
| Logistic Regression | Onto weight vector  | Separate classes  |

This project proves:

> Machine Learning = Geometric Projection + Optimization

---

#  Visualizations Included

* 3D Raw Data Scatter
* Principal Component Directions
* 3D → 2D Projection
* 2D Logistic Decision Boundary
* Projection of points onto weight vector
* Margin visualization

These visualizations make the math intuitive.

---

# Bonus Extensions

* Compare classification accuracy:

  * Before PCA
  * After PCA
* Visualize eigenvalues (variance explained)
* Show how decision boundary changes with dimensionality
* Animate projection process

---

#  What This Project Demonstrates

* Deep understanding of linear algebra in ML
* Eigen decomposition from scratch
* Logistic regression without libraries
* Geometric interpretation of ML algorithms
* Visualization-driven learning

---

#  Mathematical Foundations Used

* Vector spaces
* Mean centering
* Covariance matrices
* Eigenvalues & eigenvectors
* Orthogonal projections
* Gradient descent
* Sigmoid function

---

#  Why This Project Matters

Most ML learners use:

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
```

But this project answers:

* Why does PCA work?
* Why must we center?
* Why are eigenvectors important?
* Why is logistic regression linear?
* Why is everything projection?

This transforms ML from memorization → understanding.

---

#  Future Improvements

* Replace logistic regression with SVM
* Add Kernel PCA
* Extend to high-dimensional real datasets
* Show connection to Neural Networks (linear layers = projections)

---

#  Final Insight

Both PCA and Logistic Regression are:

> Learning the best direction in space.

One learns direction of maximum variance.

One learns direction of maximum separation.

Both are geometric.

---

