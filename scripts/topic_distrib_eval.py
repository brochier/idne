from context import idne

import idne.models.idne
import idne.eval.visualization
import idne.eval.multi_class_classification
import numpy as np
import logging
import sys
import resource
import pkg_resources
import idne.datasets.io
import sklearn.metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score


logger = logging.getLogger()
def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * 0.75, hard))

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

def main():
    dataset_name = "cora"
    adjacency_matrix, texts, labels, labels_mask = idne.datasets.io.load_multi_class_dataset(dataset_name)
    #adjacency_matrix, texts, labels, labels_mask, _ = idne.datasets.io.load_multi_label_dataset(dataset_name)

    topics = {
        0:"Case Based",
        1:"Genetic Algorithms",
        2:"Neural Networks",
        3:"Probabilistic Methods",
        4:"Reinforcement Learning",
        5:"Rule Learning",
        6:"Theory"
    }

    model = idne.models.idne.Model(number_iterations=3e3, number_incuding_points=7)
    model.fit(adjacency_matrix, texts)

    T = model.get_inducing_points()
    W = model.get_word_embeddings()
    weight = T.dot(W.T)
    num = weight > 0
    print("Ratio:", num.sum()/(weight.shape[0]*weight.shape[1]))

    alphas = model.get_topics()
    model.plot_direct_topics()
    model.plot_words_topics_amplitudes()

if __name__ == '__main__':
    memory_limit() # Limitates maximun memory usage to half
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)
