import numpy as np
import sklearn.preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import pkg_resources
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger()
from sklearn.model_selection import StratifiedShuffleSplit


def get_score(vectors, labels, proportions,n_trials=10, random_state=None):
    np.random.seed(1)
    scores = {
        'proportion': proportions,
        'micro': np.zeros(len(proportions)),
        'macro': np.zeros(len(proportions)),
        'f1': np.zeros(len(proportions)),
        'f1_std': np.zeros(len(proportions)),
        'std': np.zeros(len(proportions)),
        'c': np.zeros(len(proportions))
    }
    for i,train_ratio in enumerate(proportions):
        shuffle = StratifiedShuffleSplit(n_splits=n_trials, train_size=train_ratio, test_size=1-train_ratio, random_state=random_state)
        y = labels
        X = vectors
        std = list()
        std_f1 = list()
        for train_index, test_index in shuffle.split(X, y):
            y_train, y_test = y[train_index], y[test_index]
            X_train, X_test = X[train_index], X[test_index]
            sco_mi, sco_ma, c, f1 = train_and_predict(X_train, X_test, y_train, y_test)
            scores["micro"][i] += sco_mi/n_trials
            scores['macro'][i] += sco_ma / n_trials
            scores['f1'][i] += f1 / n_trials
            std.append(sco_mi)
            std_f1.append(f1)
            scores['c'][i] += c/n_trials
        scores["std"][i] = np.array(std).std()
        scores['f1_std'][i] = np.array(std_f1).std()
    return scores

def convert_labels(labels):
    labels_unique = np.sort(np.unique(labels))
    lab_dict = dict()
    for i, lu in enumerate(labels_unique):
        lab_dict[lu] = i
    N = labels_unique.shape[0]
    bin_lab = np.zeros((labels.shape[0], N), dtype = np.bool)
    for i, l in enumerate(labels):
        bin_lab[i,lab_dict[l]] = True

    return bin_lab

def train_and_predict(train_data, test_data, train_labels, test_labels):
    clf = LogisticRegressionCV(max_iter=1e4, class_weight='balanced')
    clf.fit(train_data, train_labels)
    predicted_labels = clf.predict_proba(test_data)
    #predicted_labels_train = clf.predict(train_data)
    scores_micro = roc_auc_score(convert_labels(test_labels), predicted_labels, average='micro')
    scores_macro = roc_auc_score(convert_labels(test_labels), predicted_labels, average='macro')

    predicted_labels = clf.predict(test_data)
    # predicted_labels_train = clf.predict(train_data)
    f1 = f1_score(test_labels, predicted_labels, average='micro')
    return scores_micro, scores_macro, clf.C_.mean(), f1


def evaluate(model, adjacency_matrix, features, labels, labels_mask, proportions, n_trials=10, random_state=None, reinit=True):
    if reinit:
        model.__init__()
    model.fit(adjacency_matrix, features)
    vectors = np.array(model.get_embeddings())[labels_mask]
    #vectors = sklearn.preprocessing.normalize(vectors, axis=0, norm='l2')
    scores = get_score(vectors, labels, proportions, n_trials=n_trials, random_state=random_state)
    return scores
