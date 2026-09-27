import pandas as pd
df=pd.read_csv('/content/income.csv')
df

#DATA UNDERSTANDING

#First 3 rows
print(df.head(3))
#last 3 rows
print(3)
#Dimension
print(df.ndim)
#Number of rowss and columns
print(df.shape)
#statistical summary
print(df.describe())
#null values
print(df.isnull().sum())
#all columns in dataset
print(df.columns)
#Non-numerical(object) columns in dataset
print(df.select_dtypes(include=object).columns)
#Numerical columns
print(df.select_dtypes(exclude=object).columns)
#Datatypes of each columns
print(df.dtypes)
#unique values count
print(df.nunique())
#target columns value count
print(df['income'].value_counts())


#DATA CLEANING

#1.null values
df.isin(['?']).sum()
df.replace('?','NAN')
print(df.isnull().sum())
#2.duplicate values
print(df.duplicated().sum())
df=df.drop_duplicates()
print(df.duplicated().sum())


#LABEL ENCODING
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
obj=df.select_dtypes(include=object).columns.to_list()
for i in obj:
  df[i]=le.fit_transform(df[i])
df

#INPUT OUTPUT SEPARATION
x=df.drop('income',axis=1)
print(x)
y=df['income']

#TRAIN AND TEST SPLITTING
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

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(xtrain1)
xtrain1=scaler.transform(xtrain1)
print(xtrain1)
xtest=scaler.transform(xtest)
xtest

#MODEL BUILDING

#1. KNN-MODEL
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
#training Model
knn.fit(xtrain1,ytrain1)
#testing model
ypred1=knn.predict(xtest)
ypred1

#2. NAIVE BAYES MODEL
from sklearn.naive_bayes import BernoulliNB
nb=BernoulliNB()
#training model
nb.fit(xtrain1,ytrain1)
#testing model
ypred2=nb.predict(xtest)
ypred2

#3.Decision tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
#training model
dt.fit(xtrain1,ytrain1)
#testing model
ypred3=dt.predict(xtest)
ypred3
















