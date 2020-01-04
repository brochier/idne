from context import idne

import idne.eval.multi_label_classification
import idne.models.idne
import idne.models.gvnrt
import idne.models.lsa
import idne.models.tadw
import idne.models.graph2gauss

import logging
import sys
import resource
import idne.datasets.io
import numpy as np


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

def main():
    dataset_name = "gaming.stackexchange.com"
    adjacency_matrix, texts, labels, labels_mask, tags = idne.datasets.io.load_multi_label_dataset(dataset_name)

    #model = idne.models.idne.Model()
    #model = idne.models.tadw.Model()
    model = idne.models.gvnrt.Model()
    #model = idne.models.lsa.Model()
    #model = idne.models.graph2gauss.Model()

    scores = idne.eval.multi_label_classification.evaluate(
        model,
        adjacency_matrix,
        texts,
        labels,
        labels_mask,
        [0.9],
        n_trials=1
    )
    print(scores)
    print("Plot")
    #model.plot_topics()
    #model.plot_direct_topics()
    #model.plot_words_topics_amplitudes()

if __name__ == '__main__':
    memory_limit() # Limitates maximun memory usage to half
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)

