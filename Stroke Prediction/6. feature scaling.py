#FEATURE SCALING

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(xtrain1)
xtrain1=scaler.transform(xtrain1)
print(xtrain1)
xtest=scaler.transform(xtest)
xtest
