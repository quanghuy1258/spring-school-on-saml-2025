lr = [1e-3, 1e-2, 1e-1, 1]
n_iter = 30
theta_init = np.ones(X.shape[1])

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1, xlabel='iteration', ylabel='loss')

for learn_rate in lr:
    _, _, vect_loss = batch_gradient_descent_MSE(X, y, theta_init, learn_rate, n_iter, vect=True)
    ax.plot(vect_loss, marker='*', alpha=0.5, label='%.03f' % learn_rate)
    
ax.set_title('Gradient descent with learning rates')
ax.set_yscale('log')
plt.legend()

plt.show()