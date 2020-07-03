import numpy as np
from sklearn.datasets import load_breast_cancer
from RandomForest import *

if __name__ == "__main__":
    data = load_breast_cancer()
    x_data = data.data
    y_data = data.target

    K = 10
    idxs = cross_validation(x_data, y_data, K)
    # n = 1~20
    # bootstrap = True
    # feature: all
    # depth: None
    for i in range(1, 20 + 1):
        total_acc = 0.
        for j in range(K):
            x_train = x_data[idxs[j][0]]
            y_train = y_data[idxs[j][0]]
            x_test = x_data[idxs[j][1]]
            y_test = y_data[idxs[j][1]]

            clf = RandomForest(n_estimators=i,
                               max_features=x_train.shape[1],
                               bootstrap=True,
                               max_depth=None)

            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            total_acc += score(y_test, y_pred)
        print('Number of tree =', i, ', acc =', total_acc/K)
    # n = 1~20
    # bootstrap = True
    # feature: 1
    # depth: None
    for i in range(1, 20 + 1):
        total_acc = 0.
        for j in range(K):
            x_train = x_data[idxs[j][0]]
            y_train = y_data[idxs[j][0]]
            x_test = x_data[idxs[j][1]]
            y_test = y_data[idxs[j][1]]

            clf = RandomForest(n_estimators=i,
                               max_features=1,
                               bootstrap=True,
                               max_depth=None)

            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            total_acc += score(y_test, y_pred)
        print('Extremely Random Forest, Number of tree =',
              i, ', acc =', total_acc/K)
