x = np.linspace(-3,3,1000)

plt.subplot(1,2,1)
plt.plot(x,f(x))
plt.xlim([-2,3])
plt.ylim([0,10])
plt.title('Function $f$ - 2 local minima')

plt.subplot(1,2,2)
plt.plot(x,g(x))
plt.xlim([-3,3])
plt.ylim([0,10])
plt.title('Function $g$ - 1 global minimum')

plt.tight_layout()
plt.show()