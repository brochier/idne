import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

import warnings
import logging
from skmultilearn.model_selection import IterativeStratification
import sklearn.preprocessing
import sklearn.model_selection
logger = logging.getLogger()

def train_and_predict(train_data, test_data, train_labels, test_labels):
    clf = MultiOutputClassifier(LogisticRegressionCV(max_iter=1e4, class_weight='balanced'))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(train_data, train_labels.A)
        y_pred = np.array(clf.predict_proba(test_data))[:,:,1].T
        mi = roc_auc_score(test_labels.A, y_pred, average="micro")
        ma = roc_auc_score(test_labels.A, y_pred, average="macro")
        c = np.mean([estimator.C_.mean() for estimator in
                          clf.estimators_])
    return mi, ma, c

def evaluate(model, adjacency_matrix, features, labels, labels_mask, proportions, n_trials = 1, random_state=None):
    scores = {
        'proportion': proportions,
        'micro': np.zeros(len(proportions)),
        'macro': np.zeros(len(proportions)),
        'std': np.zeros(len(proportions)),
        'c': np.zeros(len(proportions))
    }
    for i, train_ratio in enumerate(proportions):
        indices = np.arange(adjacency_matrix.shape[0])
        labeled_indices = indices[labels_mask]
        not_labeled_indices = np.setdiff1d(indices, labeled_indices)
        std = list()
        for _ in range(n_trials):
            stratifier = IterativeStratification(n_splits=2, order=2,
                                                 sample_distribution_per_fold=[1.0-train_ratio, train_ratio],
                                                 random_state=random_state)
            model.__init__()
            train_ind_l, test_ind_l = next(stratifier.split(labels, labels))
            train_ids_nl, _test_ids_nl = sklearn.model_selection.train_test_split(not_labeled_indices, train_size=train_ratio, test_size=1-train_ratio)
            train_ids = np.concatenate([labeled_indices[train_ind_l], train_ids_nl]) # order is important, labeled first
            test_ids = labeled_indices[test_ind_l]
            adjacency_matrix_train = adjacency_matrix[train_ids][:, train_ids].copy()
            adjacency_matrix_train.eliminate_zeros()
            features_train = [features[i] for i in train_ids]
            features_test = [features[i] for i in test_ids]
            labels_train = labels[train_ind_l]
            labels_test = labels[test_ind_l]
            model.fit(adjacency_matrix_train, features_train)
            vectors_train = np.array(model.get_embeddings_new(features_train))[:len(train_ind_l)]
            vectors_test = np.array(model.get_embeddings_new(features_test))
            logger.debug(f"train: {train_ids.shape} nodes, {labels_train.sum()} labels")
            logger.debug(f"test: {test_ids.shape} nodes, {labels_test.sum()} labels, (+{_test_ids_nl.shape} forgotten nodes)")
            logger.debug(f"adjacency: {adjacency_matrix.shape}, {adjacency_matrix_train.shape}")
            logger.debug(f"train vectors: {vectors_train.shape}, test vectors: {vectors_test.shape}")
            mi, ma, c = train_and_predict(vectors_train, vectors_test, labels_train,
                                          labels_test)
            std.append(mi)
            scores['micro'][i] += mi / n_trials
            scores['macro'][i] += ma / n_trials
            scores['c'][i] += c / n_trials
        scores['std'][i] += np.array(std).std()

    return scores