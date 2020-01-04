import numpy as np
import os
import sklearn.preprocessing
import multiprocessing
from functools import partial
import tempfile
import time
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import csgraph
import theano
from theano import tensor as T
import logging
logger = logging.getLogger()

"""
RandomWalker: class to generate sequences of nodes based on random walks from an adjacency matrix
"""
class RandomWalker:
    def __init__(self, adjacency_matrix, walks_length, walks_number):
        self.adjacency_matrix = adjacency_matrix # scipy.sparse.csr_matrix
        self.walks_length = walks_length # int
        self.walks_number = walks_number # int

    """
    build_random_walks: 
    """
    def build_random_walks(self, gather_walks = True):
        with tempfile.TemporaryDirectory() as dump_dir:
            jobs = []
            seed = int(time.time())
            random_choice = MultipleRandomChoice(self.adjacency_matrix)
            for walk_index in range(self.walks_number):
                np.random.seed(seed + walk_index) # make sure each process starts with a new seed
                func = partial(one_walk,
                               walks_length = self.walks_length,
                               random_choice = random_choice,
                               dump_dir = dump_dir)
                p = multiprocessing.Process(target=func,
                                            name=multiprocessing.current_process().name,
                                            args=(walk_index,))
                p.name = multiprocessing.current_process().name
                jobs.append(p)
                p.start()
            randoms_walks = list()
            for walk_index in range(self.walks_number):
                jobs[walk_index].join()
                if gather_walks is True:
                    logger.debug(" ".join([str(v) for v in ["Adding walk index", walk_index, "to final random walks list."]]))
                    current_random_walk = np.load(os.path.join(dump_dir, "random_walk_{0}.npy".format(walk_index)))
                    randoms_walks.append(current_random_walk)
            randoms_walks = np.vstack(randoms_walks)
            logger.debug(" ".join([str(v) for v in ["Final number of walks: ", randoms_walks.shape[0]]]))
            return randoms_walks

    def random_walks_generator(self):
        random_choice = MultipleRandomChoice(self.adjacency_matrix)
        N = random_choice.nodes_number
        for i in range(self.walks_number):
            starting_nodes = np.arange(N)
            np.random.shuffle(starting_nodes)
            for start_node in starting_nodes:
                walker_positions = np.zeros(self.walks_length + 1, dtype=np.int32)
                walker_positions[0] = start_node
                for j in range(self.walks_length):
                    next_node = random_choice[walker_positions[j]]
                    walker_positions[j+1] = next_node
                yield walker_positions

        
"""
RandomChoice: class to generate per node walk choices
"""

class RandomChoice(object):
    """
    A class to generate random choices given a probability distribution p.
    Use the same code as numpy.random.random_choice, but the initialization is made only once.
    This is useful when calling multiple times and when it is impossible to guess in advance
    the number of calls (stochastic process).
    p doesn't have to be normalized (sum!=1).
    """

    def __init__(self, p):
        p /= p.sum()
        self.cdf = p.cumsum()
        self.cdf /= self.cdf[-1]

    def get(self):
        uniform_samples = np.random.random_sample(1)
        idx = self.cdf.searchsorted(uniform_samples, side='right').astype(np.int32)
        return idx[0]

class MultipleRandomChoice(object):
    """
    A class to generate random choices given multiple probabilities distributions extracted from a sparse representation mat.
    For each row, the probability distribution is extracted. Then calling __getitem__(i) draw a sample given the distribution
    of mat[i].
    By default, if a row i has no entry, the sample i will be returned.
    """

    def __init__(self, mat):
        mat.eliminate_zeros()
        self.nodes_number = mat.shape[0]
        transition_probs = sklearn.preprocessing.normalize(mat, axis=1, norm='l1', copy=False)
        nonzero = transition_probs.nonzero()
        data = transition_probs.data
        K = len(data)
        choices = list()
        probs = list()
        k = 0
        for i in range(self.nodes_number):
            choices.append(list())
            probs.append(list())
            if k >= K:  #  if we're done looping on the data, we fill the remaining nodes probs
                choices[i] = np.array([i], dtype=np.int32)
                probs[i] = np.array([1], dtype=np.float32)
                continue
            elif nonzero[0][k] > i:  #  if the current node has no transition
                choices[i] = [i]
                probs[i] = np.array([1], dtype=np.float32)
            else:
                while k < K and nonzero[0][k] == i:  #  loop over currrent node transitions indices/probabilities
                    choices[i].append(nonzero[1][k])
                    probs[i].append(data[k])
                    k += 1
            choices[i] = np.array(choices[i], dtype=np.int32)
            probs[i] = np.array(probs[i], dtype=np.float32)
        self.choices = choices
        self.rc = list()
        for i in range(self.nodes_number):
            self.rc.append(RandomChoice(probs[i]))

    def __getitem__(self, arg):
        return self.choices[arg][self.rc[arg].get()]

"""
Create sequences of node starting RW from each node
"""
def one_walk(walk_index, walks_length, random_choice, dump_dir):
    logger.debug(" ".join([str(v) for v in ["Starting walk index", walk_index]]))
    N = random_choice.nodes_number
    walkers_positions = np.zeros((N,walks_length + 1), dtype=np.int32)
    starting_nodes = np.arange(N)
    np.random.shuffle(starting_nodes)
    for start_node in starting_nodes:
        walkers_positions[start_node,0] = start_node
        # Random walks
        for j in range(walks_length):
            next_node = random_choice[walkers_positions[start_node,j]]
            walkers_positions[start_node,j+1] = next_node
    np.save(os.path.join(dump_dir, "random_walk_{0}".format(walk_index)), walkers_positions)
    return walk_index

import cProfile, pstats, io
from pstats import SortKey
def sample_neighbors(X, max_neigh = 10):
    rows = list()
    data = list()
    cols = list()
    r, c = X.nonzero()
    d = X.data
    n = len(d)
    cursor = 0
    #pr = cProfile.Profile()
    #pr.enable()
    for i in range(X.shape[0]):
        min_v = cursor
        while cursor < n and r[cursor] == i:
            cursor+=1
        max_v = cursor
        mask = np.arange(min_v, max_v)
        top = min(max_neigh, len(mask))
        choices = c[mask]
        probs = d[mask]
        """
        cdf = probs.cumsum()
        cdf /= cdf[-1]
        uniform_samples = np.random.random_sample(top)
        indices = cdf.searchsorted(uniform_samples, side='right').astype(np.int32)
        """
        indices = np.argsort(probs)[::-1][0:top]
        rows.extend([i] * top)
        cols.extend(choices[indices].tolist())
        data.extend((probs[indices]/probs[indices].sum()).tolist())
    #pr.disable()
    #s = io.StringIO()
    #sortby = SortKey.CUMULATIVE
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=X.shape)



def pagerank_scipy(M, alpha=0.85, max_neigh=50, max_iter=10, tol=1.0e-6):
    N = M.shape[0]
    M = sklearn.preprocessing.normalize(M, 'l1', axis=1)
    X = scipy.sparse.eye(N, N, format='csr')
    for i in range(max_iter):
        Xlast = X.copy()
        X = alpha * X.dot(M)
        X += (1 - alpha) * scipy.sparse.eye(N, N, format='csr')
        X = sample_neighbors(X, max_neigh=max_neigh)
        err = scipy.absolute(X - Xlast).sum()
        print(i, err)
        if err < (N * N * tol):
            return X
    return X


def deepwalk_filter(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
    evals = np.maximum(evals, 0)
    logger.debug("After filtering, max eigenvalue=%f, min eigenvalue=%f", np.max(evals), np.min(evals))
    return evals

def approximate_normalized_graph_laplacian(A, rank, which="LA"):
    n = A.shape[0]
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} W D^{-1/2}
    X = scipy.sparse.identity(n) - L
    logger.debug("Eigen decomposition...")
    evals, evecs = scipy.sparse.linalg.eigsh(X, rank, which=which, tol=1e-3, maxiter=300)
    logger.debug("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals))
    logger.debug("Computing D^{-1/2}U..")
    D_rt_inv = scipy.sparse.diags(d_rt ** -1)
    D_rt_invU = D_rt_inv.dot(evecs)
    return evals, D_rt_invU

def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)
    X = scipy.sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    m = T.matrix()
    mmT = T.dot(m, m.T) * (vol/b)
    f = theano.function([m], T.log(T.maximum(mmT, 1)))
    Y = f(X.astype(theano.config.floatX))
    logger.debug("Computed DeepWalk matrix with %d non-zero elements", np.count_nonzero(Y))
    return scipy.sparse.csr_matrix(Y)
