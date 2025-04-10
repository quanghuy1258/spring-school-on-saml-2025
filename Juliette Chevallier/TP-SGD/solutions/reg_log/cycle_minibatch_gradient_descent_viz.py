bs = [1, 5, 100]

theta_init = np.random.rand(X_train.shape[1],1)
learn_rate = 1e-2
n_iter = 30

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1, xlabel='iteration', ylabel='loss')
for batch_size in bs:
    print('batch_size =',batch_size)
    _, _, vect_loss = cycle_minibatch_gradient_descent_BCE(X_train, y_train, theta_init, batch_size,  learn_rate, n_iter, vect=True)
    ax.plot(vect_loss, marker='.', alpha=0.5, label='$m =$%s' % batch_size)

ax.set_title('Stochastic gradient descent')
plt.legend()
plt.show()