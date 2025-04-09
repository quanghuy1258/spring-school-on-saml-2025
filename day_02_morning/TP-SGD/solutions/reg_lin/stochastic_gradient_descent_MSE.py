def stochastic_gradient_descent_MSE(X, y, theta_init, learn_rate=1e-2, n_iter=30, vect=False):
    observations_idx = np.arange(X.shape[0])
    theta = theta_init.copy()

    if not vect:
        for _ in range(n_iter):
            rd.shuffle(observations_idx)
            for idx in observations_idx:
                x_i = X[idx:idx+1, :]  # force shape (1, p)
                y_i = y[idx]           # scalaire or shape (1,)
                grad = grad_MSE(x_i, y_i, theta)
                theta -= learn_rate * grad
        return theta

    if vect:
        total_steps = n_iter * X.shape[0]
        vect_theta = np.zeros((total_steps + 1, X.shape[1]))
        vect_loss = np.zeros(total_steps + 1)

        vect_theta[0, :] = theta_init
        vect_loss[0] = mean_square_error(X, y, theta_init)

        step = 0
        for _ in range(n_iter):
            rd.shuffle(observations_idx)
            for idx in observations_idx:
                x_i = X[idx:idx+1, :]  # shape (1, p)
                y_i = y[idx]
                grad = grad_MSE(x_i, y_i, vect_theta[step, :])
                vect_theta[step+1, :] = vect_theta[step, :] - learn_rate * grad
                vect_loss[step+1] = mean_square_error(X, y, vect_theta[step+1, :])
                step += 1

        return vect_theta[-1], vect_theta, vect_loss