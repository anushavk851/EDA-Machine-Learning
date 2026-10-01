#LABEL ENCODING
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
obj=df.select_dtypes(include=object).columns
for i in obj:
  df[i]=le.fit_transform(df[i])
df

#FEATURE SELECTION
corr = df.corr(numeric_only=True)
print(corr['classification'].sort_values(ascending=False))

#INPUT OUTPUT SEPARATION
x=df.drop(['classification','su'],axis=1)
y=df['classification']
x

#TRAIN-TEST-SPLITTING
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=41)
print(xtrain.shape)
print(xtest.shape)

#DATA BALANCING
print(ytrain.value_counts())
from imblearn.over_sampling import SMOTE
sm=SMOTE()
xtrain1,ytrain1=sm.fit_resample(xtrain,ytrain)
print(ytrain1.value_counts())

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(xtrain1)
xtrain1=scaler.transform(xtrain1)
xtrain1
xtest=scaler.transform(xtest)
xtest

#MODEL BUILDING

#1 KNN MODEL BUILDING
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
#training model
knn.fit(xtrain1,ytrain1)
#testing model
ypred1=knn.predict(xtest)
ypred1

#2 NAIVE BAYES MODEL BUILDING
from sklearn.naive_bayes import BernoulliNB
nb=BernoulliNB()
#training model
nb.fit(xtrain1,ytrain1)
#testing model
ypred2=nb.predict(xtest)
ypred2

#3 DECISION TREE MODEL BUILDING
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
#training model
dt.fit(xtrain1,ytrain1)
#testing model
ypred3=dt.predict(xtest)
ypred3

#4.ENSEMBLE MODELS

#random forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=30,criterion='entropy',max_depth=3)
#training model
rfc.fit(xtrain1,ytrain1)
#testing model
ypred4=rfc.predict(xtest)
ypred4

#XGBoost
from xgboost import XGBClassifier
xgb=XGBClassifier()
#training
xgb.fit(xtrain1,ytrain1)
#testing
ypred5=xgb.predict(xtest)
ypred5










