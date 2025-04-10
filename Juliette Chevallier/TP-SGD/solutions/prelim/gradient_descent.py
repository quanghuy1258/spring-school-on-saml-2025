def gradient_descent(gradient, start, learn_rate=1e-1, n_iter=30, vect=False):

    if not vect:
        theta = start
        for _ in range(n_iter):
            theta -= learn_rate * gradient(theta)
        return theta

    if vect:
        vect_theta = np.zeros(n_iter+1)
        vect_theta[0] = start
        for i in range(n_iter):
            vect_theta[i+1] = vect_theta[i] -learn_rate * gradient(vect_theta[i])
        return vect_theta[-1], vect_theta