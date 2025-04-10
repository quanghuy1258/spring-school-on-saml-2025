def grad_MSE(X, y, theta):
    return 2 * X.T.dot(X.dot(theta) - y) / X.shape[0]


print(la.norm(grad_MSE(X, y, theta_star), 2))