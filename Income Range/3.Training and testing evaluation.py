
#TRAINING AND TESTING SCORE EVALUATION

#SCORE EVALUATION OF KNN MODEL
training_knn=knn.score(xtrain1,ytrain1)
print("KNN training score:",training_knn)
testing_knn=knn.score(xtest,ytest)
print("KNN testing score:",testing_knn)

#SCORE EVALUATION OF NAIVE BAYES MODEL
training_nb=nb.score(xtrain1,ytrain1)
print("NaiveBayes training score:",training_nb)
testing_nb=nb.score(xtest,ytest)
print("NaiveBayes testing score:",testing_nb)

#SCORE EVALUATION OF DECISION TREE MODEL
training_dt=dt.score(xtrain1,ytrain1)
print("DecisionTree training score:",training_dt)
testing_dt=dt.score(xtest,ytest)
print("DecisionTree testing score:",testing_dt)
