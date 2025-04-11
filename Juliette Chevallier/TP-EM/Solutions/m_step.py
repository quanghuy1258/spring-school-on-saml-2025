def m_step(X, T):
    """
        M step of the EM algorithm, caculates the MLE of taus, mus and sigmas
        :param X: Nxn matrix, the dataset, N number of n-dim data points
        :param T: KxN matrix, the T matrix is the posterior matrix where the i, j th component is the T_{k, i}
        :return: a 3-tuple:
            - taus: K-dim array, the estimated prior probability for each hidden variable z
            - mus: Kxn matrix, the estimated mean of the n-dim gaussian component, for each of the k component
            - sigmas: Kxnxn matrix, the covariance matrix of the n-dim gaussian component, for each of the k component
    """
    def get_sigma(X, muk, Tk):
        """
            function that calculate the covariance of the k-th component
            :param muk: n-dim vector, the k-th component's mean
            :param Tk: N-dim vector, the k-th component posterior of hidden state z_i, for each x_i
        """
        X_centred = X - muk
        X_weighted = X_centred * np.tile(Tk, reps=(X.shape[1],1)).T # repeat Tk in N-direction to match X's shape and weigh it
        return X_weighted.T@X_centred/np.sum(Tk) # weighted and centred are exchangable: we only need to weigh it by T_k once

    N, n = X.shape #  N: number of data points, n: dimension of the data point
    K = T.shape[0] # num of hidden component
    T_sum = np.sum(T, axis=1) # caculate the common term sum of T_{k, i} over all i, this is a k-dim vector

    taus = T_sum / N # average over i  for T_{k, i} gives MLE for all tau_k

    T_sum_rep = np.tile(T_sum, reps=(n, 1)).T # repeat T_sum n times in column
    mus = T@X/T_sum_rep # T@X gives a Kxn matrix with it's k, i th component be \sum_{i=1}^NT_{k, i}x_i then each row is divided by T_sum, gives MLE for all mu_k

    sigmas = np.array([get_sigma(X, mus[k, :], T[k, :]) for k in range(K)])
    return taus, mus, sigmas