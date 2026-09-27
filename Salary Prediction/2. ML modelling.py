#LABEL ENCODING
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
obj=df.select_dtypes(include=object).columns.to_list()
for i in obj:
  df[i]=le.fit_transform(df[i])
df

#INPUT OUTPUT SEPARATION
x=df.drop('Salary',axis=1)
y=df['Salary']
x

#TRAIN TEST SPLITTING
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=41)
print(xtrain.shape)
print(xtest.shape)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(xtrain)
xtrain=scaler.transform(xtrain)
print(xtrain)
xtest=scaler.transform(xtest)
print(xtest)

#MODEL BUILDING

#KNN MODEL BUILDING
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=7)
#training model
knn.fit(xtrain,ytrain)
#testing 
ypred1=knn.predict(xtest)
ypred1

#Decision tree MODEL BUILDING
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
#training model
dt.fit(xtrain,ytrain)
#testing 
ypred2=dt.predict(xtest)
ypred2
