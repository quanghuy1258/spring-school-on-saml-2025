theta_star = la.inv(X.T.dot(X)).dot(X.T).dot(y)
y_pred_star = X.dot(theta_star)

print('Vector theta_star:', theta_star)
print('Prediction', y_pred_star[:5])