__author__ = 'luoshalin'

# machine learning models

from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.svm import SVC


#     ==========/ Logistic Regression /==========
def lr(X, y):
    # TUNE
    # for C in range(1, 5, 1):
    #     lr = LogisticRegression(C=C)
    #     scores = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
    #     print scores.mean()

    lr = LogisticRegression(C=1.0)
    scores = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
    return scores.mean()


# ==========/ KNN /==========
def knn(X, y):
    knn = KNeighborsClassifier(n_neighbors=3)  # 3 is the max

    # Cross Validation & Param Selection - 3 is the max
    k_range = range(1, 31)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
    plt.plot(k_range, k_scores)
    return np.mean(k_scores)


# ==========/ GNB /==========
def gnb(X, y):
    gnb = GaussianNB()
    scores = cross_val_score(gnb, X.toarray(), y, cv=5, scoring='accuracy')
    return scores.mean()


# ==========/ MNB /==========
def mnb(X, y):
    mnb = MultinomialNB()
    scores = cross_val_score(mnb, X, y, cv=5, scoring='accuracy')
    return scores.mean()


# ==========/ BNB /==========
def bnb(X, y):
    bnb = BernoulliNB()
    scores = cross_val_score(bnb, X, y, cv=5, scoring='accuracy')
    return scores.mean()


# ==========/ Decision Tree /==========
def dtree(X, y):
    dtr = tree.DecisionTreeClassifier()
    scores = cross_val_score(dtr, X, y, cv=5, scoring='accuracy')
    return scores.mean()


# ==========/ SVM /==========
def svm(X, y):
    # TUNE
    # C_2d_range = np.arange(1, 10, 1).tolist()
    # gamma_2d_range = np.arange(0.001, 0.01, 0.001).tolist()
    # for C in C_2d_range:
    #     for gamma in gamma_2d_range:
    #         model = SVC(C=C, gamma=gamma)
    #         scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    #         print 'C = ' + str(C) + '\tgamma = ' + str(gamma) + '\t' + str(scores.mean())

    C = 7
    gamma = 0.01
    model = SVC(C=C, gamma=gamma)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean()