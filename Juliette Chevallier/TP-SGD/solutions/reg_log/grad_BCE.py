def grad_BCE(X, y, theta):
    return np.sum(-X * y / (1 + np.exp(y * X.dot(theta))), axis=0) / X.shape[0]