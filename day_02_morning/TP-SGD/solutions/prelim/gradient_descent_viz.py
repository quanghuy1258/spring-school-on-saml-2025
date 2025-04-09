x = np.linspace(-3,3,1000)

n_iter = 30
learn_rate = 1e-1

# --- #

start = -0.1
_, vect_theta_f = gradient_descent(grad_f, start, learn_rate, n_iter, vect=True)
_, vect_theta_g = gradient_descent(grad_g, start, learn_rate, n_iter, vect=True)

plt.subplot(2,2,1)
plt.plot(x,f(x))
plt.plot(vect_theta_f,f(vect_theta_f))
plt.xlim([-2,3])
plt.ylim([0,10])
plt.title('Function $f$ - 2 start = {}'.format(start))

plt.subplot(2,2,2)
plt.plot(x,g(x))
plt.plot(vect_theta_g,g(vect_theta_g))
plt.xlim([-3,3])
plt.ylim([0,10])
plt.title('Function $g$ - start = {}'.format(start))

# --- #

start = 0.1
_, vect_theta_f = gradient_descent(grad_f, start, learn_rate, n_iter, vect=True)
_, vect_theta_g = gradient_descent(grad_g, start, learn_rate, n_iter, vect=True)

plt.subplot(2,2,3)
plt.plot(x,f(x))
plt.plot(vect_theta_f,f(vect_theta_f))
plt.xlim([-2,3])
plt.ylim([0,10])
plt.title('Function $f$ - start = {}'.format(start))

plt.subplot(2,2,4)
plt.plot(x,g(x))
plt.plot(vect_theta_g,g(vect_theta_g))
plt.xlim([-3,3])
plt.ylim([0,10])
plt.title('Function $g$ - start = {}'.format(start))

# --- #

plt.tight_layout()
plt.show()