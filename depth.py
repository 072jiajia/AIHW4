import numpy as np
from sklearn.datasets import load_breast_cancer
from RandomForest import *

if __name__ == "__main__":
    data = load_breast_cancer()
    x_data = data.data
    y_data = data.target

    K = 10
    idxs = cross_validation(x_data, y_data, K)

    for i in range(1, 15 + 1):
        total_acc = 0.
        for j in range(K):
            x_train = x_data[idxs[j][0]]
            y_train = y_data[idxs[j][0]]
            x_test = x_data[idxs[j][1]]
            y_test = y_data[idxs[j][1]]

            clf = DecisionTree(max_depth=i)

            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            total_acc += score(y_test, y_pred)
        print('clf depth =', i, ', acc =', total_acc/K)

    # n = 10
    # bootstrap = True
    # feature: all
    # depth: 1 ~ 15
    for i in range(1, 15 + 1):
        total_acc = 0.
        for j in range(K):
            x_train = x_data[idxs[j][0]]
            y_train = y_data[idxs[j][0]]
            x_test = x_data[idxs[j][1]]
            y_test = y_data[idxs[j][1]]

            clf = RandomForest(n_estimators=10,
                               max_features=x_train.shape[1],
                               bootstrap=True,
                               max_depth=i)

            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            total_acc += score(y_test, y_pred)
        print('forest depth =', i, ', acc =', total_acc/K)
