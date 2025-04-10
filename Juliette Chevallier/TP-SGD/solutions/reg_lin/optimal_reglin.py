# %load solutions/reg_lin/optimal_reglin.py

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1, xlabel='x', ylabel='y')
ax.scatter(XX, yy, alpha=0.5)
ax.plot(XX, y_pred_star, color='red')
ax.set_title('Our dataset with regression line')

plt.show()