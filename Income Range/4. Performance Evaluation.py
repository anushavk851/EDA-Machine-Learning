#PERFORMANCE EVALUATION
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,recall_score,precision_score,f1_score

#A. CONFUSION MATRIX

#1.KNN
cknn=confusion_matrix(ypred1,ytest)
print(cknn)
d_knn=ConfusionMatrixDisplay(cknn)
d_knn.plot()
#2.Naive Bayes
cnb=confusion_matrix(ypred2,ytest)
print(cnb)
d_nb=ConfusionMatrixDisplay(cnb)
d_nb.plot()
#3.Decision Tree
cdt=confusion_matrix(ypred3,ytest)
print(cdt)
d_dt=ConfusionMatrixDisplay(cdt)
d_dt.plot()

#ACCURACY SCORE
print("Accuracy of KNN:",accuracy_score(ypred1,ytest))
print("Accuracy of NaiveBayes:",accuracy_score(ypred2,ytest))
print("Accuracy of DecisionTree:",accuracy_score(ypred3,ytest))

#PRECISION SCORE
print("Precision of KNN:",precision_score(ypred1,ytest))
print("Precision of NaiveBayes:",precision_score(ypred2,ytest))
print("Precision of DecisionTree:",precision_score(ypred3,ytest))

#RECALL SCORE
print("Recall of KNN:",recall_score(ypred1,ytest))
print("Recall of NaiveBayes:",recall_score(ypred2,ytest))
print("Recall of DecisionTree:",recall_score(ypred3,ytest))

#F1 SCORE
print("F1 score of KNN:",f1_score(ypred1,ytest))
print("F1 score of NaiveBayes:",f1_score(ypred2,ytest))
print("F1 score of DecisionTree:",f1_score(ypred3,ytest))
