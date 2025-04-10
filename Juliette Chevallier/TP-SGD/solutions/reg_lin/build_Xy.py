n = X.shape[0]
X = np.concatenate([np.ones((n, 1)), XX], axis=1)
y = yy

print('Matrix X:\n', X[:5,])
print('Vector y:', y[:5,])