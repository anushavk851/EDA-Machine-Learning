#7 Performance Evaluation

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

#KNN
mae = mean_absolute_error(ytest, ypred1)
mse = mean_squared_error(ytest, ypred1)
rmse = np.sqrt(mse)
r2 = r2_score(ytest, ypred1)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

#DECISION TREE
mae = mean_absolute_error(ytest, ypred2)
mse = mean_squared_error(ytest, ypred2)
rmse = np.sqrt(mse)
r2 = r2_score(ytest, ypred2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
