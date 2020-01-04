import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

import warnings
import logging
from skmultilearn.model_selection import iterative_train_test_split
import sklearn.preprocessing
logger = logging.getLogger()

def train_and_predict(X, y, train_ratio=0.2, n_trials=10, random_state=None):
    micro, macro, c, std, f1, f1_std = [], [], [], [], [], []
    for i in range(n_trials):
        np.random.seed(random_state)
        X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=1-train_ratio)
        clf = MultiOutputClassifier(LogisticRegressionCV(max_iter=1e4, class_weight='balanced'))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train.A)
            y_pred = np.array(clf.predict_proba(X_test))[:,:,1].T
            mi = roc_auc_score(y_test.A,y_pred, average="micro")
            ma = roc_auc_score(y_test.A,y_pred, average="macro")
            y_pred = clf.predict(X_test)
            f = f1_score(y_test.A, y_pred, average="micro")
        std.append(mi)
        f1.append(f)
        f1_std.append(f)
        micro.append(mi)
        macro.append(ma)
        c.append(np.mean([estimator.C_.mean() for estimator in
                   clf.estimators_]))
    return np.mean(micro), np.mean(macro), np.mean(c), np.std(std), np.mean(f1), np.std(f1_std)

def get_score(vectors, labels, proportions, n_trials = 10, random_state=None):
    np.random.seed(1)
    scores = {
        'proportion': proportions,
        'micro':np.zeros(len(proportions)),
        'macro':np.zeros(len(proportions)),
        'std': np.zeros(len(proportions)),
        'f1': np.zeros(len(proportions)),
        'f1_std': np.zeros(len(proportions)),
        'c': np.zeros(len(proportions))
    }
    for i,p in enumerate(proportions):
        scores["micro"][i], scores["macro"][i], scores['c'][i], scores['std'][i], scores["f1"][i], scores["f1_std"][i] = train_and_predict(vectors, labels, train_ratio=p, n_trials=n_trials, random_state=random_state)
    return scores

def evaluate(model, adjacency_matrix, texts, labels, labels_mask, proportions, random_state=None, n_trials=10):
    model.__init__()
    model.fit(adjacency_matrix, texts)
    emb = model.get_embeddings()
    vectors = np.array(emb)[labels_mask]
    #vectors = sklearn.preprocessing.normalize(vectors, axis=0, norm='l2')
    scores = get_score(vectors, labels, proportions, n_trials=n_trials, random_state=random_state)
    return scores