#MODEL Building

#1 KNN Model Building
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
#training model
knn.fit(xtrain1,ytrain1)
#testing model
ypred1=knn.predict(xtest)
ypred1

#2 Naive Bayes Model Building
from sklearn.naive_bayes import BernoulliNB
nb=BernoulliNB()
#training model
nb.fit(xtrain1,ytrain1)
#testing model
ypred2=nb.predict(xtest)
ypred2

#3 Decision Tree model Building
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
#training model
dt.fit(xtrain1,ytrain1)
#testing model
ypred3=dt.predict(xtest)
ypred3
