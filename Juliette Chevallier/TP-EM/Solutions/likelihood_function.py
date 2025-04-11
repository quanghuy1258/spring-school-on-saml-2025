def likelihood_function(X, taus, mus, sigmas):
    """
    The component pdf p_k(x; mu_k, sigma_k)
        :param taus: K-dim array contains the weight (or prior of hidden var) of each gaussian component
        :param mus: Kxn array (n be the dimension of x vector and K be the num of components), mean of k n-dim gaussian distr.
        :param sigmas: Knxn array covariance of k n-dim gaussian distr.
        :return: numeric between 0 and 1, the likelihood function value
    """
    N = X.shape[0] # number of data points
    get_component_prob = lambda x: component_pdfs(x, mus, sigmas)
    T = np.apply_along_axis(arr=X, func1d=get_component_prob, axis=1) # gaussian component probabilities in row format (NxK)
    taus_rep = np.tile(taus, reps=(N, 1)) # repeat tau along N-axis so elementwise product can work

    return np.sum(T*taus_rep, axis=1)