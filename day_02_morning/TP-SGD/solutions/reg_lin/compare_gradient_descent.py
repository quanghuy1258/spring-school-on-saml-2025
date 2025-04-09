#theta_init = np.random.randn(X.shape[1])
theta_init = np.ones(X.shape[1])
#theta_init = np.mean(X, axis=0)

n_iter=50
learn_rate=1e-3

theta_GD, _, vect_loss_GD = batch_gradient_descent_MSE(X, y, theta_init, learn_rate, n_iter, vect=True)
theta_SGD, _, vect_loss_SGD = stochastic_gradient_descent_MSE(X, y, theta_init, learn_rate, n_iter, vect=True)
idx_SGD = np.arange(1, n_iter*X.shape[0]+1, X.shape[0])
idx_SGD = np.insert(idx_SGD, 0, 0)

print('> LSE computation vs GD vs SGD:')
print('theta_star', theta_star)
print('theta obtained by GD', theta_GD)
print('theta obtained by SGD', theta_SGD)
print('')

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1, xlabel='iteration', ylabel='loss')
ax.plot(vect_loss_GD, color='blue', marker='*', alpha=0.5, label='GD')
ax.plot(vect_loss_SGD[idx_SGD], color='red', marker='*', alpha=0.5, label='SGD')

ax.set_title('Gradient descent')
plt.legend()
plt.show()