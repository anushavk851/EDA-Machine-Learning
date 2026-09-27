#PERFORMANCE EVALUATION OF EACH
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,recall_score,precision_score,f1_score

#A. Confusion Matrix

#KNN
cknn=confusion_matrix(ypred1,ytest)
print(cknn)
d_knn=ConfusionMatrixDisplay(cknn)
d_knn.plot()
#Naive Bayes
cnb=confusion_matrix(ypred2,ytest)
print(cnb)
d_nb=ConfusionMatrixDisplay(cnb)
d_nb.plot()
#Decision tree
cdt=confusion_matrix(ypred3,ytest)
print(cdt)
d_dt=ConfusionMatrixDisplay(cdt)
d_dt.plot()


#B. Accuracy score
a_knn=accuracy_score(ypred1,ytest)
print("KNN:",a_knn)
a_nb=accuracy_score(ypred2,ytest)
print("Naive Bayes:",a_nb)
a_dt=accuracy_score(ypred3,ytest)
print("Decision Tree:",a_dt)


#C. precision score
p_knn=precision_score(ypred1,ytest)
print("KNN:",p_knn)
p_nb=precision_score(ypred2,ytest)
print("Naive Bayes:",p_nb)
p_dt=precision_score(ypred3,ytest)
print("Decision Tree:",p_dt)


#D. Recall score
r_knn=recall_score(ypred1,ytest)
print("KNN:",r_knn)
r_nb=recall_score(ypred2,ytest)
print("Naive Bayes:",r_nb)
r_dt=recall_score(ypred3,ytest)
print("Decision Tree:",r_dt)


#E. F1 score
f_knn=f1_score(ypred1,ytest)
print("KNN:",f_knn)
f_nb=f1_score(ypred2,ytest)
print("Naive Bayes:",f_nb)
f_dt=f1_score(ypred3,ytest)
print("Decision Tree:",f_dt)


