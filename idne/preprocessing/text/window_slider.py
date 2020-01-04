import numpy as np
import scipy.sparse

import logging
logger = logging.getLogger()

"""
WindowSlider: Class to generate co-occurrences count by sliding a window on sequences
"""
class WindowSlider:
    def __init__(self,
                 corpus, # callable generator or iterator
                 nodes_number, # int
                 window_size = 10, # size of the window (on one side => 3 => [1/3, 1/2, 1, X, 1, 1/2, 1/3])
                 window_type = "harmonic", # "flat" [1,1,X,1,1] or "harmonic" [1/2, 1, X, 1, 1/2]
                 rw_size_limit = 10000000 # memory limit when building co-occurrence matrix
                 ):

        self.corpus = corpus
        self.window_size = window_size
        self.window_type = window_type
        self.nodes_number = nodes_number
        self.rw_size_limit = rw_size_limit
        self.steps_count = len(corpus)

    def build_cooccurrence_matrix(self):
        mat = scipy.sparse.csr_matrix((self.nodes_number, self.nodes_number))
        dat = list()
        row_ind = list()
        col_ind = list()
        percentage = 0
        step = 0
        for i,sequence in enumerate(self.corpus):
            step +=1
            if (step/self.steps_count) > percentage:
                logger.debug("Window Slider Progression: %s", "{0:.2f}".format(percentage))
                percentage+=0.01
            sequence_length = len(sequence)
            if len(sequence) < 2 or sequence[1] == -1: # if no neighbor
                continue
            for l in range(sequence_length):
                word = sequence[l]
                w_size = np.random.randint(1, self.window_size+1)
                window = [w for w in sequence[max(l - w_size, 0):min(l + w_size + 1, sequence_length)]]
                pos = l - max(l - w_size, 0)
                for m, w in enumerate(window):
                    if m != pos:
                        factor = None
                        if self.window_type == "harmonic":
                            factor = 1 / (abs(m - pos))
                        elif self.window_type == "uniform":
                            factor = 1
                        dat.append(factor)
                        row_ind.append(word)
                        col_ind.append(w)

            if len(dat) > self.rw_size_limit:
                logger.debug("rw_size_limit reached !")
                mat += scipy.sparse.csr_matrix((np.array(dat),
                                                (np.array(row_ind),
                                                 np.array(col_ind))),
                                               shape=(self.nodes_number,
                                                      self.nodes_number))
                dat = list()
                row_ind = list()
                col_ind = list()
        mat += scipy.sparse.csr_matrix((np.array(dat), (np.array(row_ind),
                                                        np.array(col_ind))),
                                       shape=(self.nodes_number,
                                              self.nodes_number))
        logger.debug( "Cooccurrence matrix density={0:.2f}% mass={1:.2f}".format(
                len(mat.data) * 100 / (self.nodes_number * self.nodes_number),
                mat.sum()
              ))
        return mat
