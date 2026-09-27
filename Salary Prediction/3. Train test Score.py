#TRAINING AND TESTING SCORE

#KNN
print("training score for knn:",knn.score(xtrain,ytrain))
print("testing score for knn:",knn.score(xtest,ytest))
#DECISION TREE
print("training score for Decision Tree:",dt.score(xtrain,ytrain))
print("testing score for Decision Tree:",dt.score(xtest,ytest))
