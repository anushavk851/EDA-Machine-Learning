#TRAINING AND TESTING SCORE

#KNN
print("KNN training score",knn.score(xtrain1,ytrain1))
print("KNN testing score",knn.score(xtest,ytest))
#NAIVE BAYES
print("Naive training score",nb.score(xtrain1,ytrain1))
print("Naive testing score",nb.score(xtest,ytest))
#DECISION TREE
print("tree training score",dt.score(xtrain1,ytrain1))
print("tree testing score",dt.score(xtest,ytest))

#PERFORMANCE EVALUATION
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score

#A.CONFUSION MATRIX
#KNN
cknn=confusion_matrix(ypred1,ytest)
print(cknn)
d_knn=ConfusionMatrixDisplay(cknn)
d_knn.plot()
#NAIVE BAYES
cnb=confusion_matrix(ypred2,ytest)
print(cnb)
d_nb=ConfusionMatrixDisplay(cnb)
d_nb.plot()
#DECISION TREE
cdt=confusion_matrix(ypred3,ytest)
print(cdt)
d_dt=ConfusionMatrixDisplay(cdt)
d_dt.plot()

#B.ACCURACY SCORE
print("accuracy of KNN:",accuracy_score(ypred1,ytest))
print("accuracy of Naive Bayes:",accuracy_score(ypred2,ytest))
print("accuracy of Decision Tree:",accuracy_score(ypred3,ytest))

#C.PRECISION SCORE
print("precision score of KNN:",precision_score(ypred1,ytest))
print("precision score of Naive Bayes:",precision_score(ypred2,ytest))
print("precision score of Decision Tree:",precision_score(ypred3,ytest))

#D. RECALL SCORE
print("recall score of KNN:",recall_score(ypred1,ytest))
print("recall score of Naive Bayes:",recall_score(ypred2,ytest))
print("recall score of Decision Tree:",recall_score(ypred3,ytest))

#E. F1 SCORE
print("f1 score of KNN:",f1_score(ypred1,ytest))
print("f1 score of Naive Bayes:",f1_score(ypred2,ytest))
print("f1 score of Decision Tree:",f1_score(ypred3,ytest))
