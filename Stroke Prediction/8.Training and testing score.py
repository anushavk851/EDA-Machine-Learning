#TRAINING AND TESTING SCORE EVALUATION

#1. KNN model Evaluation
#training score
training_score=knn.score(xtrain1,ytrain1)
print(training_score)
#testing score
testing_score=knn.score(xtest,ytest)
print(testing_score)

#2. Naive bayes model Evaluation
#training score
training_score=nb.score(xtrain1,ytrain1)
print(training_score)
#testing score
testing_score=nb.score(xtest,ytest)
print(testing_score)

#3. Decision tree model Evaluation
#training score
training_score=dt.score(xtrain1,ytrain1)
print(training_score)
#testing score
testing_score=dt.score(xtest,ytest)
print(testing_score)
