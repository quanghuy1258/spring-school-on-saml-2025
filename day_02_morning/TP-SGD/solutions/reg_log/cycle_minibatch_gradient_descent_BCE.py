def cycle_minibatch_gradient_descent_BCE(X, y, theta_init, batch_size, learn_rate=1e-2, n_iter=30, vect=False):
    observations_idx = np.arange(X.shape[0])
    rd.shuffle(observations_idx)
    
    theta = theta_init.copy()
    n, p  = X.shape

    if not vect:
        for _ in range(n_iter):
            for j in range(n):
                start = j * batch_size % n
                end = min(start + batch_size, n)
                idx = np.arange(start, end)
                idx = observations_idx[idx]

                X_batch = X[idx,:]
                y_batch = y[idx]   
                grad = grad_BCE(X_batch, y_batch, theta)
                grad = grad[:, np.newaxis]
                theta -= learn_rate * grad
        return theta

    if vect:
        total_steps = n_iter * n
        vect_theta = np.zeros((total_steps + 1, p))
        vect_loss = np.zeros(total_steps + 1)

        vect_theta[0, :] = theta.squeeze()
        vect_loss[0] = binary_cross_entropy(X, y, theta_init)

        step = 0
        for _ in range(n_iter):
            for j in range(n):
                start = j * batch_size % n
                end = min(start + batch_size, n)
                idx = np.arange(start, end)
                idx = observations_idx[idx]
                
                X_batch = X[idx,:]
                y_batch = y[idx]
                grad = grad_BCE(X_batch, y_batch, theta)
                grad = grad[:, np.newaxis]
                theta -= learn_rate * grad
                
                vect_theta[step+1, :] = theta.squeeze()
                vect_loss[step+1] = binary_cross_entropy(X, y, theta)
                step += 1

        return vect_theta[-1], vect_theta, vect_loss