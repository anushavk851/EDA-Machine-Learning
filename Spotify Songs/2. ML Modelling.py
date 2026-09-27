#LABEL ENCODING
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
obj=df.select_dtypes(include=object).columns.to_list()
for i in obj:
  df[i]=le.fit_transform(df[i])
df

#INPUT OUTPUT SEPARATION
x=df.drop(['artist(s)_name','mode','track_name',],axis=1)
y=df['mode']
print(y)

#TRAIN TEST SPLITTING
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

#KNN MODEL BUILDING
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
#training model
knn.fit(xtrain1,ytrain1)
#testing model
ypred1=knn.predict(xtest)
ypred1

#NAIVE BAYES MODEL BUILDING
from sklearn.naive_bayes import BernoulliNB
nb=BernoulliNB()
#training model
nb.fit(xtrain1,ytrain1)
#testing model
ypred2=nb.predict(xtest)
ypred2

#DECISION TREE MODEL BUILDING
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
#training model
dt.fit(xtrain1,ytrain1)
#testing model
ypred3=dt.predict(xtest)
ypred3





