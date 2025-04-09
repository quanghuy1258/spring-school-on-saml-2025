def binary_cross_entropy(X, y, theta):
    return np.sum(np.log(1 + np.exp(-y * X.dot(theta)))) / X.shape[0]