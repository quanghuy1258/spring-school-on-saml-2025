def e_step(X, taus, mus, sigmas):
    """
        E step of the EM algorithm, caculates the posterior T_{k, i}=P(z_i=k|y_i)
        it returns T_{k,i} in the form of a KxN T matrix where each element is T_{k, i}
        :param X: Nxn matrix represents N number of n-dim data points
        :param taus: K-dim vector, the weight of each component, or the prior of the hidden stats z
        :param mus: Kxn matrix (n be the dimension of x vector and K be the num of components), mean of k n-dim gaussian distr.
        :param sigmas: Kxnxn matrix covariance of k n-dim gaussian distr.
        :return: T_{k,i} in the form of a KxN T matrix where each element is T_{k, i}
    """
    K, N = mus.shape[0], X.shape[0] # dimensions, K: num of hidden component, N: number of data points
    get_component_prob = lambda x: component_pdfs(x, mus, sigmas)
    T = np.apply_along_axis(arr=X, func1d=get_component_prob, axis=1) # gaussian component probabilities in row format (NxK)
    taus_rep = np.tile(taus, reps=(N, 1)) # repeat tau along N-axis so elementwise product can work

    norm_const = np.sum(T*taus_rep, axis=1) # the normalisation factor \sum_{k=1}^K p_k * tau_k， and is currently estimated likelihood
    norm_const_rep = np.tile(norm_const, reps=(K, 1)).T # repeat normalisation constant along K-axis

    T = T*taus_rep/norm_const_rep # calculate the posterior 
    return T.T #return the transposed matrix so that the index is matched