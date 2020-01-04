import numpy as np
import sklearn.preprocessing
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
import pkg_resources
from sklearn.metrics import roc_auc_score

import logging

logger = logging.getLogger()
from sklearn.model_selection import StratifiedShuffleSplit

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
    scores_mi = roc_auc_score(convert_labels(test_labels), predicted_labels, average='micro')
    scores_ma = roc_auc_score(convert_labels(test_labels), predicted_labels, average='macro')
    return scores_mi, scores_ma, clf.C_.mean()


def evaluate(model, adjacency_matrix, features, labels, labels_mask, proportions,  n_trials = 1, random_state=None):
    scores = {
        'proportion': proportions,
        'micro': np.zeros(len(proportions)),
        'macro': np.zeros(len(proportions)),
        'std': np.zeros(len(proportions)),
        'c': np.zeros(len(proportions))
    }
    for i, train_ratio in enumerate(proportions):
        shuffle = StratifiedShuffleSplit(n_splits=n_trials, train_size=train_ratio, test_size=1-train_ratio,
                                         random_state=random_state)
        std = list()
        for train_ids, test_ids in shuffle.split(features, labels):
            model.__init__()
            adjacency_matrix_train = adjacency_matrix[train_ids][:, train_ids].copy()
            adjacency_matrix_train.eliminate_zeros()
            features_train = [features[i] for i in train_ids]
            features_test = [features[i] for i in test_ids]
            labels_mask_train = labels_mask[train_ids]
            labels_train = labels[train_ids][labels_mask_train]
            labels_mask_test = labels_mask[test_ids]
            labels_test = labels[test_ids][labels_mask_test]
            model.fit(adjacency_matrix_train, features_train)
            vectors_train = np.array(model.get_embeddings_new(features_train))[labels_mask_train]
            vectors_test = np.array(model.get_embeddings_new(features_test))[labels_mask_test]
            mi, ma, c = train_and_predict(vectors_train, vectors_test, labels_train,
                                          labels_test)
            std.append(mi)
            scores['micro'][i] += mi / n_trials
            scores['macro'][i] += ma / n_trials
            scores['c'][i] += c / n_trials
        scores["std"][i] = np.array(std).std()

    return scores
