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

    ax.set_title('3D Scatter Plot of Raw Synthetic Data')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')  
    ax.set_zlabel('Feature 3')
    ax.legend()
    plt.show()

    '''
    As the projection is in 3D so the method scatter has (x,y,z) coordinate
    and here we have X[y==0][:,0] which is the x coordinate of class 0 
    i.e. all rows of 1st columns and likewise we have y and z 
    
    '''

def plot_pca(X,y,eigenvectors,eigenvalues,mean):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # plotting the clusters initially
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], X[y == 0][:, 2], color='blue', label='Class 0', alpha=0.5)
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], X[y == 1][:, 2], color='red', label='Class 1', alpha=0.5)

    # drawing each principal component as an arrow
    colors = ['green', 'orange', 'purple']

    for i in range(3): # we have 3 principal components as we have 3 features

        arrow_length = np.sqrt(eigenvalues[i])*2.5
        direction = eigenvectors[:, i] * arrow_length

        ax.quiver(
            mean[0], mean[1], mean[2],        # arrow starts at mean of data
            direction[0], direction[1], direction[2],  # arrow points in PC direction
            color=colors[i],
            linewidth=3,
            label=f'PC{i+1} (variance={eigenvalues[i]:.2f})'
        )

    ax.set_title('PCA Principal Component Directions')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.legend()
    plt.show()

    '''
    in this function, there is an important part which is:
    eigenvectors[:, i] returns the i-th principal component direction as a unit vector.
    and as unit vector doesnot describe the variance along the direction 
    so we scale up using the square root of the corresponding eigenvalue to get the actual length of the arrow representing the variance along that principal component.

    '''

def plot_3d_vs_2d(X, y, X_train_pca, y_train):

    fig = plt.figure(figsize=(14, 6))

    # original datasets in left
    ax1 = fig.add_subplot(121, projection='3d')  
    ax1.scatter(X[y == 0][:, 0], X[y == 0][:, 1], X[y == 0][:, 2],
                color='blue', label='Class 0', alpha=0.4)
    ax1.scatter(X[y == 1][:, 0], X[y == 1][:, 1], X[y == 1][:, 2],
                color='red', label='Class 1', alpha=0.4)
    ax1.set_title('Original 3D Data')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('X3')
    ax1.legend()

    # PCA reduced 2D data in right
    ax2 = fig.add_subplot(122) 
    ax2.scatter(X_train_pca[y_train == 0][:, 0], X_train_pca[y_train == 0][:, 1],
                color='blue', label='Class 0', alpha=0.4)
    ax2.scatter(X_train_pca[y_train == 1][:, 0], X_train_pca[y_train == 1][:, 1],
                color='red', label='Class 1', alpha=0.4)
    ax2.set_title('PCA Reduced 2D Data')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend()

    plt.suptitle('3D → 2D Projection via PCA', fontsize=14) 
    plt.tight_layout()
    plt.show()



def plot_decision_boundary(X_train_pca, y_train, model):

    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the 2D points
    ax.scatter(X_train_pca[y_train == 0][:, 0], X_train_pca[y_train == 0][:, 1],
               color='blue', label='Class 0', alpha=0.4)
    ax.scatter(X_train_pca[y_train == 1][:, 0], X_train_pca[y_train == 1][:, 1],
               color='red', label='Class 1', alpha=0.4)

    # build the decision boundary line
    x1_min, x1_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    x1_values = np.linspace(x1_min, x1_max, 100)

    # solve for x2: x2 = -(w1*x1 + b) / w2
    x2_values = -(model.weights[0] * x1_values + model.bias) / model.weights[1]

    ax.plot(x1_values, x2_values, color='black', linewidth=2, label='Decision Boundary')

    ax.set_title('Logistic Regression Decision Boundary')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    plt.show()




def plot_weight_projection(X_train_pca, y_train, model):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # weight vector direction on 2D scatter - left
    ax1.scatter(X_train_pca[y_train == 0][:, 0], X_train_pca[y_train == 0][:, 1],
                color='blue', label='Class 0', alpha=0.4)
    ax1.scatter(X_train_pca[y_train == 1][:, 0], X_train_pca[y_train == 1][:, 1],
                color='red', label='Class 1', alpha=0.4)

    # draw weight vector as arrow from origin
    w = model.weights
    ax1.annotate('', 
                xy=(w[0]*2, w[1]*2),        
                xytext=(0, 0),               
                arrowprops=dict(color='black', lw=2, arrowstyle='->')
    )
    ax1.text(w[0]*2.2, w[1]*2.2, 'w', fontsize=14, color='black')
    ax1.set_title('Weight Vector Direction')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()
    ax1.axhline(0, color='gray', linewidth=0.5) 
    ax1.axvline(0, color='gray', linewidth=0.5)  

    # 1D projection of all points onto w - right
    # project each point onto w using dot product
    w_unit = w / np.linalg.norm(w)  
    projections = X_train_pca @ w_unit  

    ax2.scatter(projections[y_train == 0], np.zeros(sum(y_train == 0)),
                color='blue', label='Class 0', alpha=0.4)
    ax2.scatter(projections[y_train == 1], np.zeros(sum(y_train == 1)),
                color='red', label='Class 1', alpha=0.4)

    # draw the decision threshold
    threshold = -model.bias / np.linalg.norm(w)
    ax2.axvline(threshold, color='black', linewidth=2, label='Decision Threshold')

    ax2.set_title('1D Projection onto Weight Vector')
    ax2.set_xlabel('Projection value')
    ax2.set_yticks([])  # hide y axis, everything is on a line
    ax2.legend()

    plt.suptitle('Logistic Regression as Projection', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_margin(X_train_pca, y_train, model):

    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the 2D points
    ax.scatter(X_train_pca[y_train == 0][:, 0], X_train_pca[y_train == 0][:, 1],
               color='blue', label='Class 0', alpha=0.4)
    ax.scatter(X_train_pca[y_train == 1][:, 0], X_train_pca[y_train == 1][:, 1],
               color='red', label='Class 1', alpha=0.4)

    # build x1 range
    x1_min, x1_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    x1_values = np.linspace(x1_min, x1_max, 100)

    # decision boundary:        w1*x1 + w2*x2 + b = 0
    # upper margin boundary:    w1*x1 + w2*x2 + b = 1
    # lower margin boundary:    w1*x1 + w2*x2 + b = -1

    x2_boundary = -(model.weights[0] * x1_values + model.bias) / model.weights[1]
    x2_upper    = -(model.weights[0] * x1_values + model.bias - 1) / model.weights[1]
    x2_lower    = -(model.weights[0] * x1_values + model.bias + 1) / model.weights[1]

    # draw the three lines
    ax.plot(x1_values, x2_boundary, color='black', linewidth=2, label='Decision Boundary')
    ax.plot(x1_values, x2_upper, color='green', linewidth=1.5, linestyle='--', label='Upper Margin')
    ax.plot(x1_values, x2_lower, color='green', linewidth=1.5, linestyle='--', label='Lower Margin')

    # shade the margin region between the two dashed lines
    ax.fill_between(x1_values, x2_lower, x2_upper, alpha=0.1, color='green', label='Margin Region')

    ax.set_title('Decision Boundary with Margin')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    plt.show()



def plot_accuracy_comparison(X_train, X_test, y_train, y_test,
                              X_train_pca, X_test_pca):
    
    from logistic_regression import logisticregression

    # --- model 1: trained on raw 3D data ---
    model_raw = logisticregression(learning_rate=0.1, n_iters=2000)
    model_raw.fit(X_train, y_train)
    predictions_raw = model_raw.prediction(X_test)
    accuracy_raw = np.mean(predictions_raw == y_test)

    # --- model 2: trained on PCA reduced 2D data ---
    model_pca = logisticregression(learning_rate=0.1, n_iters=2000)
    model_pca.fit(X_train_pca, y_train)
    predictions_pca = model_pca.prediction(X_test_pca)
    accuracy_pca = np.mean(predictions_pca == y_test)

    # --- plot bar chart ---
    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        ['Raw 3D Data', 'PCA 2D Data'],
        [accuracy_raw, accuracy_pca],
        color=['steelblue', 'coral'],
        width=0.4
    )

    # add accuracy value on top of each bar
    for bar, acc in zip(bars, [accuracy_raw, accuracy_pca]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x position = center of bar
            bar.get_height() + 0.01,             # y position = just above bar
            f'{acc:.3f}',                         # accuracy rounded to 3 decimals
            ha='center', fontsize=12
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy: Before vs After PCA')
    plt.tight_layout()
    plt.show()

    print(f"Raw 3D Accuracy:  {accuracy_raw:.3f}")
    print(f"PCA 2D Accuracy:  {accuracy_pca:.3f}")


def plot_variance_explained(eigenvalues):

    total_variance = np.sum(eigenvalues)
    variance_ratio = eigenvalues / total_variance * 100  # convert to percentage

    # cumulative variance — how much is captured by PC1+PC2, PC1+PC2+PC3 etc
    cumulative_variance = np.cumsum(variance_ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- LEFT: individual variance per component ---
    bars = ax1.bar(
        [f'PC{i+1}' for i in range(len(eigenvalues))],
        variance_ratio,
        color=['steelblue', 'coral', 'lightgreen']
    )

    # add percentage on top of each bar
    for bar, var in zip(bars, variance_ratio):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{var:.1f}%',
            ha='center', fontsize=12
        )

    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('Variance Explained per Component')
    ax1.set_ylim(0, 100)

    # --- RIGHT: cumulative variance ---
    ax2.plot(
        [f'PC{i+1}' for i in range(len(eigenvalues))],
        cumulative_variance,
        marker='o', color='steelblue', linewidth=2, markersize=8
    )

    # add percentage next to each point
    for i, var in enumerate(cumulative_variance):
        ax2.text(i, var + 1.5, f'{var:.1f}%', ha='center', fontsize=12)

    # draw a horizontal line at 95% — common threshold for keeping components
    ax2.axhline(95, color='red', linestyle='--', linewidth=1.5, label='95% threshold')

    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Cumulative Variance Explained')
    ax2.set_ylim(0, 110)
    ax2.legend()

    plt.suptitle('PCA Variance Analysis', fontsize=14)
    plt.tight_layout()
    plt.show()

    # print the numbers clearly
    for i, (var, cum) in enumerate(zip(variance_ratio, cumulative_variance)):
        print(f"PC{i+1}: {var:.1f}% variance  |  cumulative: {cum:.1f}%")