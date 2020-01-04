import numpy as np
import sklearn.metrics
import scipy.sparse
import logging

logger = logging.getLogger()

"""
ROC AUC score
"""


def get_roc_auc_score(y_true, y_score):
    """
    Compute ROC AUC given true labels and score predictions
    :param y_true: [0,1,1,1,0,1,0,0,,1,0,1,0,1,0,1,0,1,...]
    :param y_score: [0.4,0.9,0.5,0.778,0.152,0.123,0.6,...]
    :return: Float representing the area under the ROC curve
    """
    #print("PREDICTIONS (min,max,mean,std): ", (y_score.min(), y_score.max(), y_score.mean(), y_score.std()))
    return sklearn.metrics.roc_auc_score(y_true, y_score, average='micro')

def make_symetric(X):
    X.eliminate_zeros()
    X.sum_duplicates()
    rows, cols = X.nonzero()
    data = X.data
    pairs_set = set()
    pairs_list = list()
    pairs_data = list()
    for i, (r,c) in enumerate(zip(rows,cols)):
        if (r,c) not in pairs_set and (c,r) not in pairs_set:
            pairs_set.add((r,c))
            pairs_list.append((r,c))
            pairs_data.append(data[i])

    new_rows = np.array([val[0] for val in pairs_list], dtype=np.int)
    new_cols = np.array([val[1] for val in pairs_list], dtype=np.int)
    new_data = np.array(pairs_data, dtype=np.float)

    a = scipy.sparse.csr_matrix(
        (
            np.concatenate((new_data, new_data)),
            (np.concatenate((new_rows, new_cols)),
            np.concatenate((new_cols, new_rows)))
        ),
        shape=X.shape)

    a.setdiag(0)
    a.sum_duplicates()
    a.eliminate_zeros()
    return a

def generate_test_set_edges(adjacency_matrix, ratio, random_state=None):
    """
    Produces train/test sets for link prediction, by sampling edges
    :param adjacency_matrix:
    :param text_sequences:
    :param ratio:
    :return:
    """

    adjacency_matrix_train = scipy.sparse.triu(adjacency_matrix).copy().tocsr()
    adjacency_matrix_train.eliminate_zeros()
    length_data = len(adjacency_matrix_train.data)
    np.random.seed(random_state)
    keep_mask = np.random.choice(a=[False, True], size=length_data, p=[1 - ratio, ratio])
    x_test = list()
    y_true = list()
    x_train = list()
    y_true_train = list()
    nonzeros = adjacency_matrix_train.nonzero()
    num_nodes = adjacency_matrix_train.shape[0]
    non_zer_sets = dict()
    for i,j in zip(*adjacency_matrix_train.nonzero()):
        non_zer_sets.setdefault(i, set())
        non_zer_sets.setdefault(j, set())
        non_zer_sets[i].add(j)
        non_zer_sets[j].add(i)
    for k, (i, j) in enumerate(zip(nonzeros[0], nonzeros[1])):
        if not keep_mask[k]:
            x_test.append((i, j))
            y_true.append(True)
            false_i = i
            false_j = np.random.randint(0, num_nodes, dtype=np.int32)
            while false_i == false_j or false_j in non_zer_sets[false_i]:
                false_j = np.random.randint(0, num_nodes, dtype=np.int32)
            x_test.append((false_i, false_j))
            y_true.append(False)
        else:
            x_train.append((i, j))
            y_true_train.append(True)
            false_i = i
            false_j = np.random.randint(0, adjacency_matrix_train.shape[0], dtype=np.int32)
            while false_i == false_j or false_j in non_zer_sets[false_i]:
                false_j = np.random.randint(0, adjacency_matrix_train.shape[0], dtype=np.int32)
            x_train.append((false_i, false_j))
            y_true_train.append(False)
    adjacency_matrix_train.data = adjacency_matrix_train.data * keep_mask
    adjacency_matrix_train.eliminate_zeros()
    logger.debug("Link prediction with p={1} test set generated with {0}"
                " pairs of nodes and train set with {2} pairs of nodes".format(len(x_test), ratio,
                                                                               len(x_train)))
    y_true = np.array(y_true, dtype=np.bool)
    y_true_train = np.array(y_true_train, dtype=np.bool)
    x_test = np.array(x_test, dtype=np.int)
    x_train = np.array(x_train, dtype=np.int)
    return make_symetric(adjacency_matrix_train), x_test, y_true, x_train, y_true_train


def test(model, x_test, y_true):
    y_score = list()
    for k, (i, j) in enumerate(x_test):
        y_score.append(model.predict(i, j))
    return get_roc_auc_score(y_true, np.nan_to_num(np.array(y_score, dtype=np.float)))


def evaluate(model, adjacency_matrix, features, proportions, n_trials = 10, random_state=None):
    scores = {
        'proportion': proportions,
        'micro': np.zeros(len(proportions)),
        'std': np.zeros(len(proportions)),
        'micro_train': np.zeros(len(proportions))
    }
    for i, p in enumerate(proportions):
        std = list()
        for _ in range(n_trials):
            model.__init__()
            adjacency_matrix_train, x_test, y_true, x_train, y_true_train = \
                generate_test_set_edges(
                    adjacency_matrix,
                    p,
                    random_state
                )

            model.fit(adjacency_matrix_train, features)

            subsampling = np.arange(len(x_test))
            np.random.shuffle(subsampling)
            subsampling = subsampling[0:1000]

            sc = test(
                model,
                x_test[subsampling],
                y_true[subsampling]
            )

            scores["micro"][i] += sc/n_trials

            std.append(sc)

            """
            subsampling = np.arange(len(x_train))
            np.random.shuffle(subsampling)
            subsampling = subsampling[0:1000]

            scores["micro_train"][i] += test(
                model,
                x_train[subsampling], # TODO: find another way
                y_true_train[subsampling]
            )/n_trials
            """
        scores["std"][i] = np.array(std).std()
    return scores
