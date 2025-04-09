def batch_gradient_descent_MSE(X, y, theta_init, learn_rate=1e-2, n_iter=30, vect=False):
    theta = theta_init.copy()

    if not vect:
        for _ in range(n_iter):
            grad = grad_MSE(X, y, theta)
            theta -= learn_rate * grad
        return theta

    if vect:
        vect_theta = np.zeros((n_iter + 1, X.shape[1]))
        vect_loss = np.zeros(n_iter + 1)

        vect_theta[0, :] = theta
        vect_loss[0] = mean_square_error(X, y, theta)

        for i in range(n_iter):
            grad = grad_MSE(X, y, vect_theta[i])
            vect_theta[i+1, :] = vect_theta[i, :] - learn_rate * grad
            vect_loss[i+1] = mean_square_error(X, y, vect_theta[i+1, :])

        return vect_theta[-1], vect_theta, vect_loss