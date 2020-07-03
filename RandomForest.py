import numpy as np
from sklearn.datasets import load_breast_cancer


def cross_validation(x_train, y_train, k=5):
    '''generate cross validation train/test sets' indices'''
    Idx = np.arange(len(x_train))
    np.random.shuffle(Idx)

    Folds = []
    basic_size = len(x_train) // k
    remain_size = len(x_train) % k

    for firstn in range(remain_size):
        low = firstn * basic_size + firstn
        high = (firstn + 1) * basic_size + firstn + 1
        Folds.append(Idx[low: high])

    for lastn in range(remain_size, k):
        low = lastn * basic_size + remain_size
        high = (lastn + 1) * basic_size + remain_size
        Folds.append(Idx[low: high])

    Splits = []
    for i in range(k):
        train = []
        test = []
        for j in range(k):
            if i == j:
                test.extend(Folds[j])
            else:
                train.extend(Folds[j])
        Splits.append([np.array(train), np.array(test)])

    return np.array(Splits)


def gini(sequence):
    '''compute the gini index of a sequence'''
    length = len(sequence)
    elements = set(sequence)
    total_gini = 1
    for element in elements:
        proportion = (sequence == element).sum() / length
        total_gini -= proportion * proportion

    return total_gini


def impurity_2class(left, right):
    ''' compute the impurity of an array whose numbers
    of elements has been counted.
    In this hw, I split the array to two smaller array,
    and all the classes of the objects have either 0 or 1
    so I just make this function to compute the impurity easier.
    '''
    leftsize = left[0] + left[1]
    rightsize = right[0] + right[1]
    Totalsize = leftsize + rightsize
    pL0 = left[0]/leftsize
    pL1 = left[1]/leftsize
    pR0 = right[0]/rightsize
    pR1 = right[1]/rightsize

    return 1 - ((leftsize / Totalsize) * (pL0 * pL0 + pL1 * pL1) +
                (rightsize / Totalsize) * (pR0 * pR0 + pR1 * pR1))


def get_attr_best_split(X, Y, attr):
    ''' for data X, Y, and the attribute's index 'attr'
    compute the best boundary to split X, Y to
    x1, y1, x2, y2 which have smaller impurity.
    '''
    # I sort it first
    rearrange_index = X[:, attr].argsort()
    sorted_X = X[rearrange_index]
    sorted_Y = Y[rearrange_index]

    # initialize the variables which will be updated and return
    least_impurity = float('inf')
    boundary = None

    # this is the number of the elements in the left array
    # and the right array, it can make my function run faster
    # because I just have to check the class of current object
    # and +-1 the value of array 'left' and 'right', not to split my array
    left = [0, 0]
    right = [(sorted_Y == 0).sum(), (sorted_Y == 1).sum()]

    # for idx from 0 ~ len(X) - 1, if
    idx = 0
    while idx < len(sorted_X)-1:
        # update array 'left' and 'right'
        left[sorted_Y[idx]] += 1
        right[sorted_Y[idx]] -= 1

        # if objects whose index is idx and idx + 1 have the same value,
        # continue
        if sorted_X[idx][attr] == sorted_X[idx + 1][attr]:
            idx += 1
            continue

        # get the impurity and update the
        # boundary and least_impurity if it's smaller
        impurity = impurity_2class(left, right)
        if impurity < least_impurity:
            least_impurity = impurity
            boundary = (sorted_X[idx][attr] + sorted_X[idx + 1][attr])/2

        idx += 1

    return least_impurity, boundary


def best_split(X, Y, Max_feature=None):
    '''Get the best attribute's index and boundary'''

    # initialize the variables which will be updated and return
    least_impurity = float('inf')
    best_attr = None
    best_boundary = None

    # attrs is a range or a numpy.ndarray which contains
    # the candidate attribute's indices
    if Max_feature is None:
        attrs = range(X.shape[1])
    else:
        attrs = np.random.choice(X.shape[1], Max_feature, replace=False)

    # for every attribute's index in attrs, get it's smallest
    # impurity and best boundary and then update the return variables
    for attr in attrs:
        impurity, boundary = get_attr_best_split(X, Y, attr)

        if impurity < least_impurity:
            least_impurity = impurity
            best_attr = attr
            best_boundary = boundary

    return best_attr, best_boundary


def split_data(X, Y, attr, boundary):
    # split X, Y into lx, ly, rx, ry by boundary
    l_index = X[:, attr] <= boundary
    lx = X[l_index]
    ly = Y[l_index]

    r_index = X[:, attr] > boundary
    rx = X[r_index]
    ry = Y[r_index]

    return lx, ly, rx, ry


class DecisionTreeNode:
    '''This is a class for Dicision Tree's Node.
    - attribute: attribute's index
    - boudary: when X[attr] <= boundary,
            go to left childnode, else, right childnode
    - impurity: impurity of this node
    - samples: the number of samples of this node
    - value: the number of sample which is class 0
            and class 1 in value[0], value[1]
    - CLASS: CLASS = 0 when value[0] > value[1], else, CLASS = 1
    - lchild, rchild: childnodes
    '''

    def __init__(self):
        self.attribute = None
        self.boundary = None
        self.impurity = None
        self.samples = None
        self.value = None
        self.CLASS = None

        self.lchild = None
        self.rchild = None

        return None

    def fit(self, X, Y, max_depth):
        '''fit X, Y using recursive method'''
        # compute the data of this node
        self.impurity = gini(Y)
        self.value = [(Y == 0).sum(), (Y == 1).sum()]
        self.samples = self.value[0] + self.value[1]
        self.CLASS = 0 if self.value[0] > self.value[1] else 1

        # if it is the leaf of the node or it contains
        # only one class (impurity == 0), return
        if max_depth == 0 or self.impurity == 0:
            return

        # get the best attribute and boundary
        self.attribute, self.boundary = best_split(X, Y)

        # if boundary == None --> we can't split the data anymore, return
        if self.boundary is None:
            return

        # split the X, Y by self.attribute and self.boundary
        LX, LY, RX, RY = split_data(X, Y, self.attribute, self.boundary)

        # make childnode
        self.lchild = DecisionTreeNode()
        self.rchild = DecisionTreeNode()
        depth = None if max_depth is None else max_depth - 1
        self.lchild.fit(LX, LY, depth)
        self.rchild.fit(RX, RY, depth)

        return None

    def predict_one(self, x):
        '''predict one data'''
        # if it's leaf node return self.CLASS
        # else, return childnode
        if self.lchild is None:
            return self.CLASS
        if x[self.attribute] <= self.boundary:
            return self.lchild.predict_one(x)
        else:
            return self.rchild.predict_one(x)


class DecisionTree(DecisionTreeNode):
    '''A decision tree classifier
    - max_depth: max depth of the tree, if max_depth is None
                divide until we can't split until no more change
    '''

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.features = None
        return None

    def fit(self, x_batch, y_batch):
        '''fit x_batch, y_batch'''
        super().fit(x_batch, y_batch, self.max_depth)

    def predict(self, x_batch):
        '''predict x_batch's data's classes'''
        X = np.array(x_batch)
        y_pred = []
        for x in X:
            y_pred.append(super().predict_one(x))
        return np.array(y_pred)


def score(a, b):
    ''' a simple function to check the proportion
    of the same data in a and b'''
    Count = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            Count += 1
    return Count/len(a)


def bagging(x_train, y_train):
    bagging_index = np.random.randint(0, len(x_train), len(x_train))
    X = x_train[bagging_index]
    Y = y_train[bagging_index]
    return X, Y


class RandomForest():
    '''A Random Forest
    - trees: decision trees of the forest
    - max_feature: the maximum number of feature when choosing attribute
    - bootstrap: do bagging or not
    '''

    def __init__(self, n_estimators, max_features,
                 bootstrap=True, max_depth=None):
        self.trees = [DecisionTree(max_depth=max_depth)
                      for i in range(n_estimators)]
        self.max_features = max_features
        self.bootstrap = bootstrap
        return None

    def fit(self, x_train, y_train):
        '''fit x_train, y_train'''
        for tree in self.trees:
            if self.bootstrap:
                X, Y = bagging(x_train, y_train)
            else:
                X, Y = x_train, y_train
            tree.fit(X, Y)

    def predict(self, x_test):
        '''predict the class of the test data'''
        predicts = np.array([tree.predict(x_test) for tree in self.trees])
        y_pred = []
        for row in predicts.T:
            # Use majority votes to get the final prediction
            if (row == 0).sum() < (row == 1).sum():
                y_pred.append(1)
            else:
                y_pred.append(0)

        return np.array(y_pred)
