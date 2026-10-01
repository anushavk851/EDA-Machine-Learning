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
#RANDOM FOREST
print("Random Forest training score",rfc.score(xtrain1,ytrain1))
print("Random Forest testing score",rfc.score(xtest,ytest))
#XGBoost
print("XGBOOST training score",xgb.score(xtrain1,ytrain1))
print("XGBOOST testing score",xgb.score(xtest,ytest))

#PERFORMANCE EVALUATION
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score

#A.CONFUSION MATRIX

#KNN
cknn=confusion_matrix(ytest,ypred1)
print(cknn)
d_knn=ConfusionMatrixDisplay(cknn)
d_knn.plot()
#NAIVE BAYES
cnb=confusion_matrix(ytest,ypred2)
print(cnb)
d_nb=ConfusionMatrixDisplay(cnb)
d_nb.plot()
#Decision Tree
cdt=confusion_matrix(ytest,ypred3)
print(cdt)
d_dt=ConfusionMatrixDisplay(cdt)
d_dt.plot()
#Random Forest
crfc=confusion_matrix(ytest,ypred4)
print(crfc)
d_rfc=ConfusionMatrixDisplay(crfc)
d_rfc.plot()
#XGBoost
cxgb=confusion_matrix(ytest,ypred5)
print(cxgb)
d_xgb=ConfusionMatrixDisplay(cxgb)
d_xgb.plot()


#B.ACCURACY SCORE
print("KNN accuracy:",accuracy_score(ytest,ypred1))
print("NB accuracy:",accuracy_score(ytest,ypred2))
print("DT accuracy:",accuracy_score(ytest,ypred3))
print("RFB accuracy:",accuracy_score(ytest,ypred4))
print("XGB accuracy:",accuracy_score(ytest,ypred5))

#C.RECALL SCORE
print("KNN Recall:",recall_score(ytest,ypred1))
print("NB Recall:",recall_score(ytest,ypred2))
print("DT Recall:",recall_score(ytest,ypred3))
print("RFB Recall:",recall_score(ytest,ypred4))
print("XGB Recall:",recall_score(ytest,ypred5))

#D.PRECISION SCORE
print("KNN precision:",precision_score(ytest,ypred1))
print("NB precision:",precision_score(ytest,ypred2))
print("DT precision:",precision_score(ytest,ypred3))
print("RFB precision:",precision_score(ytest,ypred4))
print("XGB precision:",precision_score(ytest,ypred5))

#E.F1 SCORE
print("KNN f1_score:",f1_score(ytest,ypred1))
print("NB f1_score:",f1_score(ytest,ypred2))
print("DT f1_score:",f1_score(ytest,ypred3))
print("RFB f1_score:",f1_score(ytest,ypred4))
print("XGB f1_score:",f1_score(ytest,ypred5))


#tree visualization
import matplotlib.pyplot as plt
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(dt,feature_names=x.columns,
              class_names=df['classification'].astype(str).unique(),
              filled=True)
plt.show()














