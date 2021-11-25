import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
import random
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# This function plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix(test_y, predict_y):

    C = confusion_matrix(test_y, predict_y)  
    A =(((C.T)/(C.sum(axis=1))).T)
    B =(C/C.sum(axis=0))
 
    
    labels = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
    
    #print("-"*20, "Confusion matrix", "-"*20)
    fig1,ax = plt.subplots(figsize=(20,7))
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    

    #print("-"*20, "Precision matrix (Columm Sum=1)", "-"*20)
    fig2,ax = plt.subplots(figsize=(20,7))
    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    
    #print("-"*20, "Recall matrix (Row sum=1)", "-"*20)
    fig3,ax = plt.subplots(figsize=(20,7))
    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    
    return fig1,fig2,fig3

def defining_model(model_name,i):
    if model_name=="KNN":
        clf = KNeighborsClassifier(n_neighbors=i)
    elif model_name=="Logistic Regression":
        clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    elif model_name=="SVM":
        clf = SGDClassifier(alpha=i, penalty='l2', loss='hinge', random_state=42)
    return clf
    

def training(alpha,model,X_tr,y_train,X_cr,y_cv,X_te,y_test):
    cv_log_error_array = []
    for i in alpha:
        print("for alpha =", i)
        clf = defining_model(model,i)
        clf.fit(X_tr, y_train)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(X_tr, y_train)
        sig_clf_probs = sig_clf.predict_proba(X_cr)
        cv_log_error_array.append(log_loss(y_cv, sig_clf_probs, labels=clf.classes_, eps=1e-15))
        print("Log Loss :",log_loss(y_cv, sig_clf_probs)) 

    fig, ax = plt.subplots()
    ax.plot(np.log10(alpha), cv_log_error_array,c='g')
    for i, txt in enumerate(np.round(cv_log_error_array,3)):
        ax.annotate((alpha[i],str(txt)), (np.log10(alpha[i]),cv_log_error_array[i]))
    plt.grid()
    plt.xticks(np.log10(alpha))
    plt.title("Cross Validation Error for each alpha")
    plt.xlabel("Alpha i's")
    plt.ylabel("Error measure")
    
    best_alpha = np.argmin(cv_log_error_array)
    clf = defining_model(model,alpha[best_alpha])
    clf.fit(X_tr, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(X_tr,y_train)


    predict_y = sig_clf.predict_proba(X_tr)
    print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(X_cr)
    print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(X_te)
    print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

    y_pred = sig_clf.predict(X_te)
    print("Number of mis-classified points :", np.count_nonzero((y_pred- y_test))/y_test.shape[0])
    confusion,precision,recall = plot_confusion_matrix(y_test, y_pred)
    return fig,confusion,precision,recall,sig_clf

def rf_training(X_tr,y_train,X_cr,y_cv,X_te,y_test):
    alpha = [100,200,500,1000,2000]
    max_depth = [5, 10]
    cv_log_error_array = []
    for i in alpha:
        for j in max_depth:
            print("for n_estimators =", i,"and max depth = ", j)
            clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)
            clf.fit(X_tr, y_train)
            sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
            sig_clf.fit(X_tr, y_train)
            sig_clf_probs = sig_clf.predict_proba(X_cr)
            cv_log_error_array.append(log_loss(y_cv, sig_clf_probs, labels=clf.classes_, eps=1e-15))
            print("Log Loss :",log_loss(y_cv, sig_clf_probs)) 


    best_alpha = np.argmin(cv_log_error_array)
    clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/len(max_depth))], criterion='gini', max_depth=max_depth[int(best_alpha%len(max_depth))], random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(X_tr,y_train)


    predict_y = sig_clf.predict_proba(X_tr)
    print('For values of best estimator = ', alpha[int(best_alpha/len(max_depth))], ' and best max depth = ',max_depth[int(best_alpha%len(max_depth))], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(X_cr)
    print('For values of best estimator = ', alpha[int(best_alpha/len(max_depth))], ' and best max depth = ',max_depth[int(best_alpha%len(max_depth))], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(X_te)
    print('For values of best estimator = ', alpha[int(best_alpha/len(max_depth))], ' and best max depth = ',max_depth[int(best_alpha%len(max_depth))], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

    y_pred = sig_clf.predict(X_te)
    print("Number of mis-classified points :", np.count_nonzero((y_pred- y_test))/y_test.shape[0])

    confusion,precision,recall = plot_confusion_matrix(y_test, y_pred)
    return confusion,precision,recall,sig_clf

def model_tuning(X_tr,y_train,X_cr,y_cv,X_te,y_test):
    print("\nTraining the preprocessed dataset using the model : KNN\n")
    print("Hyperparameter tuning on the model........................!")
    alpha = [5, 11, 15, 21, 31, 41, 51, 99]
    plot_knn,confusion_knn,precision_knn,recall_knn,model_knn = training(alpha,"KNN",X_tr,y_train,X_cr,y_cv,X_te,y_test)

    print("\nTraining the preprocessed dataset using the model : Logistic Regression\n")
    print("Hyperparameter tuning on the model........................!")
    alpha = [10 ** x for x in range(-6, 3)]
    plot_lr,confusion_lr,precision_lr,recall_lr,model_lr = training(alpha,"Logistic Regression",X_tr,y_train,X_cr,y_cv,X_te,y_test)

    print("\nTraining the preprocessed dataset using the model : SVM\n")
    print("Hyperparameter tuning on the model........................!")
    alpha = [10 ** x for x in range(-6, 3)]
    plot_svm,confusion_svm,precision_svm,recall_svm,model_svm = training(alpha,"SVM",X_tr,y_train,X_cr,y_cv,X_te,y_test)

    print("\nTraining the preprocessed dataset using the model : Random Forest\n")
    print("Hyperparameter tuning on the model........................!")
    alpha = [10 ** x for x in range(-6, 3)]
    confusion_rf,precision_rf,recall_rf,model_rf = rf_training(X_tr,y_train,X_cr,y_cv,X_te,y_test)

    return plot_knn,confusion_knn,precision_knn,recall_knn,model_knn,\
           plot_lr,confusion_lr,precision_lr,recall_lr,model_lr,\
           plot_svm,confusion_svm,precision_svm,recall_svm,model_svm,\
           confusion_rf,precision_rf,recall_rf,model_rf


