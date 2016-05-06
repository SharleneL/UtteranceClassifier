__author__ = 'luoshalin'

import sys
from data_process import get_train_data
from models import lr, knn, gnb, bnb, mnb, dtree, svm


def main(argv):
    param_fpath = sys.argv[1]
    # prepare training data
    X, y = get_train_data(param_fpath)  # X: feature list  y: label list

    # train
    print '\nBegin Running Logistic Regression...'
    lr_acc = lr(X, y)
    print "LR Accuracy: " + str(lr_acc)

    print '\nBegin Running KNN...'
    knn_acc = knn(X, y)
    print "KNN Accuracy: " + str(knn_acc)

    print '\nBegin Running GNB...'
    gnb_acc = gnb(X, y)
    print "GNB Accuracy: " + str(gnb_acc)

    print '\nBegin Running MNB...'
    mnb_acc = mnb(X, y)
    print "MNB Accuracy: " + str(mnb_acc)

    print '\nBegin Running BNB...'
    bnb_acc = bnb(X, y)
    print "BNB Accuracy: " + str(bnb_acc)

    print '\nBegin Running Decision Tree...'
    dtree_acc = dtree(X, y)
    print "Decision Tree Accuracy: " + str(dtree_acc)

    print '\nBegin Running SVM...'
    svm_acc = svm(X, y)
    print "SVM Accuracy: " + str(svm_acc)


if __name__ == '__main__':
    main(sys.argv)