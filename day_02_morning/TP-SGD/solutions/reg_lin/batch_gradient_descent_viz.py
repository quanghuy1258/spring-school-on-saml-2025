n_iter=30
learn_rate=1e-1

theta_init = np.random.randn(X.shape[1])
#theta_init = np.ones(X.shape[1])
#theta_init = np.mean(X, axis=0)

theta, vect_theta, vect_loss = batch_gradient_descent_MSE(X, y, theta_init, learn_rate, n_iter, vect=True)

print('> LSE computation vs GD:')
print('theta_star', theta_star)
print('theta obtained by GD', theta)
print('')

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1, xlabel='iteration', ylabel='loss')
ax.plot(vect_loss, color='blue', marker='*', alpha=0.5)
ax.set_title('Gradient descent')

plt.show()