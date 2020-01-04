import numpy as np
import scipy.sparse
import itertools
from multiprocessing.pool import ThreadPool

import logging
logger = logging.getLogger()

"""
RandomSkip: class to draw sub sampling probabilities 
"""
class RandomSkip(object):
    def __init__(self, subsampling_prob):
        self.nodes_number = len(subsampling_prob)
        self.subsampling_prob = subsampling_prob

    def __getitem__(self, arg):
        # The smaller self.subsampling_prob[arg], the more chance False is returned
        return np.random.uniform(0, 1.0) < self.subsampling_prob[arg]

"""
Group elements from an iterator
"""
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

"""
WindowSlider: Class to generate co-occurrences count by sliding a window on sequences
"""
class WindowSlider:
    def __init__(self,
                 corpus, # callable generator or iterator
                 nodes_number, # int
                 neg_sampling = False, # boolean: if generating a negative sampling
                 subsampling_threshold = 0.001, # 10e-3 to 10e-6 when lots of nodes
                 k_neg = 0, # number of negative samples to draw
                 n_epochs = 1, # number f passes over the data
                 window_size = 1, # size of the window (on one side => 3 => [1/3, 1/2, 1, X, 1, 1/2, 1/3])
                 window_factor = "flat", # "flat" [1,1,X,1,1] or "decreasing" [1/2, 1, X, 1, 1/2]
                 rw_size_limit = 10000000 # memory limit when building co-occurrence matrix
                 ):
        self.corpus = corpus
        self.window_size = window_size
        self.window_factor = window_factor
        self.nodes_number = nodes_number
        self.rw_size_limit = rw_size_limit
        self.neg_sampling = neg_sampling
        self.nodes_negsampling_prob = None # 1 prob per node
        self.nodes_subsampling_prob = None # 1 prob per node
        self.subsampling_threshold = subsampling_threshold
        self.k_neg = k_neg
        self.n_epochs = n_epochs
        self.steps_count = 0 # Number of sequences to deal with
        if self.neg_sampling == True:
            occurrences_counts = np.ones(self.nodes_number+1) # last value reserved for -1 node (no neighbor)
            for i, sequence in enumerate(self.get_corpus()):
                self.steps_count += 1
                if len(sequence) < 2 or sequence[1] == -1:  # if no neighbor
                    continue
                sequence_length = len(sequence)
                for l in range(sequence_length):
                    node = sequence[l]
                    occurrences_counts[node] +=1
            # Prob to be drawn = 1/(p**0.75) where p is global probability
            flatten_probs =  np.power(1/occurrences_counts[:-1], 0.75)
            self.nodes_negsampling_prob = flatten_probs /  flatten_probs.sum()
            f = occurrences_counts[:-1]/occurrences_counts[:-1].sum()
            t = self.subsampling_threshold
            # small values are more likely to be skipped
            # high occurrences counts produce small values
            self.nodes_subsampling_prob = (np.sqrt(f/t)+1)*t/f
            self.nodes_subsampling_prob[self.nodes_subsampling_prob > 1] = 1
            logger.debug(
                "Min=%s/Max=%s/Mean=%s/Std=%s/Sum=%s nodes_neg_sampling_prob:",
                self.nodes_negsampling_prob.min(),
                self.nodes_negsampling_prob.max(),
                self.nodes_negsampling_prob.mean(),
                self.nodes_negsampling_prob.std(),
                self.nodes_negsampling_prob.sum()
            )
            logger.debug(
                "Min=%s/Max=%s/Mean=%s/Std=%s/Sum=%s nodes_sub_sampling_prob:",
                self.nodes_subsampling_prob.min(),
                self.nodes_subsampling_prob.max(),
                self.nodes_subsampling_prob.mean(),
                self.nodes_subsampling_prob.std(),
                self.nodes_subsampling_prob.sum()
            )
        else:
            self.nodes_subsampling_prob = np.ones(self.nodes_number) / self.nodes_number
            for i, sequence in enumerate(self.get_corpus()):
                self.steps_count += 1
        logger.debug(" ".join([str(v) for v in ["Corpus length estimation: ", self.steps_count]]))

    """
    Get corpus depending on its type: init the generator if the case
    """
    def get_corpus(self):
        if callable(self.corpus):
            return self.corpus()
        else:
            return self.corpus

    """
    build_cooccurrence_matrix: create co-occurrence matrix 
    """
    def build_cooccurrence_matrix(self):
        mat = scipy.sparse.csr_matrix((self.nodes_number, self.nodes_number))
        dat = list()
        row_ind = list()
        col_ind = list()
        percentage = 0
        step = 0
        for i,sequence in enumerate(self.get_corpus()):
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
                        if self.window_factor == "decreasing":
                            factor = 1 / abs(m - pos)
                        elif self.window_factor == "flat":
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
        logger.debug( "Cooccurrence matrix density=%s mass=%s ",
                len(mat.data) * 100 / (self.nodes_number * self.nodes_number),
                mat.sum()
              )
        mat.sum_duplicates()
        mat.eliminate_zeros()
        return mat

    def generate_pairs(self, num_threads, n):
        skipper = RandomSkip(self.nodes_subsampling_prob)
        target_count = 0
        context_count = 0
        current_step = 0
        for sequences in grouper(self.get_corpus(), num_threads):
            data_list = list()
            current_step += num_threads
            step_ratio = (n*self.steps_count+current_step) / (self.n_epochs * self.steps_count)
            sequences = [seq for seq in sequences if seq is not None and len(seq) > 2 and seq[1] != -1]  # if no neighbor
            if not sequences:
                continue
            for i in range(len(sequences)):
                sequences[i] = [s for s in sequences[i] if skipper[s]]
            seq_len = [len(seq) for seq in sequences]
            seq_indices = list() # seq_indices = [0,1,2,3,0,1,2,3,0,1,2,0,2,2]
            seq_orders = list()  # seq_orders = [0,0,0,0,1,1,1,1,2,2,2,3,3,4]
            for i in range(max(seq_len)):
                for j in range(len(sequences)):
                    if i < seq_len[j]:
                        seq_indices.append(j)
                        seq_orders.append(i)
            for seq_i, seq_o in zip(seq_indices, seq_orders):
                sequence = sequences[seq_i]
                sequence_length = len(sequence)
                l = seq_o
                word = sequence[l]
                w_size = np.random.randint(1, self.window_size + 1)
                window = [w for w in
                          sequence[max(l - w_size, 0):min(l + w_size + 1, sequence_length)]]
                pos = l - max(l - w_size, 0)
                for m, w in enumerate(window):
                    if m != pos:
                        neg_samples = None
                        if self.neg_sampling is True and self.k_neg > 0:
                            neg_samples = np.random.choice(np.arange(self.nodes_number),
                                                           size=self.k_neg,
                                                           p=self.nodes_negsampling_prob)
                        factor = None
                        if self.window_factor == "decreasing":
                            factor = 1 / (abs(m - pos))
                        elif self.window_factor == "flat":
                            factor = 1

                        data = {
                            "target": word,
                            "context": w,
                            "factor": factor,
                            "neg_sampling": False,
                            "target_count": target_count,
                            "context_count": context_count,
                            "epoch": n,
                            "step_ratio": step_ratio
                        }
                        data_list.append(data)
                        target_count += 1
                        for w_neg in neg_samples:
                            data = {
                                "target": word,
                                "context": w_neg,
                                "factor": factor,
                                "neg_sampling": True,
                                "target_count": target_count,
                                "context_count": context_count,
                                "epoch": n,
                                "step_ratio": step_ratio
                            }
                            data_list.append(data)
                            context_count += 1
            #  TODO: Smart yield thread by thread
            np.random.shuffle(data_list)
            yield data_list



    def async_sequences(self,num_threads, n):
        it = self.generate_pairs(num_threads, n)
        pairs = next(it)
        pool = ThreadPool(processes=1)
        while pairs:
            async_result = pool.apply_async(it.__next__) # Start computing next it while yielding current one
            yield pairs
            pairs = async_result.get()

    """
    Generate pairs of nodes like 
    d = {
            "target": [int],             # target node
            "context": [int],            # context node
            "factor": [float],           # factor of the window
            "neg_sampling": [boolean],   # if the pair is drawn from neg sampling
            "target_count": [int],       # number of targets pushed
            "context_count": [int],      # number of contexts pushed
            "epoch": [int],              # current epoch number
            "step_ratio": [float]        # progression from 0 to 1
        }
    """
    def cooccurrence_generator(self, num_threads = 1):
        for n in range(self.n_epochs):
            for data_list in self.async_sequences(num_threads, n):
                for d in data_list:
                    yield d


