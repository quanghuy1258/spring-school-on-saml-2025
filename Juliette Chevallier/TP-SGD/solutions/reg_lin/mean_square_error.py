def mean_square_error(X, y, theta):
    diff = X.dot(theta) - y
    return diff.dot(diff) / X.shape[0]

mse = mean_square_error(X, y, theta_star)
print('MSE: %.02f RMSE: %.02f' % (mse, np.sqrt(mse)))