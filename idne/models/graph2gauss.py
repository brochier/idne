import numpy as np
import tensorflow as tf
from .graph2gauss_utils import *
from sklearn.preprocessing import normalize
import idne.preprocessing.text.dictionary
import idne.preprocessing.text.vectorizers
from sklearn.decomposition import TruncatedSVD
import scipy.spatial.distance
import numpy as np
import scipy.sparse.linalg
import logging
logger = logging.getLogger()



class Graph2Gauss:
    """
    Implementation of the method proposed in the paper:
    'Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking'
    by Aleksandar Bojchevski and Stephan Günnemann,
    published at the 6th International Conference on Learning Representations (ICLR), 2018.
    Copyright (C) 2018
    Aleksandar Bojchevski
    Technical University of Munich
    """
    def __init__(self, A, X, L, K=1, n_hidden=None, max_iter=2000, tolerance=100, seed=0):
        """
        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Sparse unweighted adjacency matrix
        X : scipy.sparse.spmatrix
            Sparse attribute matirx
        L : int
            Dimensionality of the node embeddings
        K : int
            Maximum distance to consider
        n_hidden : list(int)
            A list specifying the size of each hidden layer, default n_hidden=[512]
        max_iter :  int
            Maximum number of epoch for which to run gradient descent
        tolerance : int
            Used for early stopping. Number of epoch to wait for the score to improve on the validation set
        seed : int
            Random seed used to split the edges into train-val-test set

        """
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.saved_vars = {}

        self.X = tf.sparse_placeholder(tf.float32)
        self.feed_dict = {self.X: sparse_feeder(X)}


        self.N, self.D = X.shape
        self.L = L
        self.max_iter = max_iter
        self.tolerance = tolerance


        if n_hidden is None:
            n_hidden = [512]
        self.n_hidden = n_hidden

        hops = get_hops(A, K)

        scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                           hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                       for h in hops}

        self.__build()
        self.__dataset_generator(hops, scale_terms)
        self.__build_loss()

    def __build(self):
        w_init = tf.contrib.layers.xavier_initializer

        sizes = [self.D] + self.n_hidden

        encoded = None

        for i in range(1, len(sizes)):
            W = tf.get_variable(name='W{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float32,
                                initializer=w_init())
            b = tf.get_variable(name='b{}'.format(i), shape=[sizes[i]], dtype=tf.float32, initializer=w_init())

            if i == 1:
                encoded = tf.sparse_tensor_dense_matmul(self.X, W) + b
            else:
                encoded = tf.matmul(encoded, W) + b

            encoded = tf.nn.relu(encoded)

        W_mu = tf.get_variable(name='W_mu', shape=[sizes[-1], self.L], dtype=tf.float32, initializer=w_init())
        b_mu = tf.get_variable(name='b_mu', shape=[self.L], dtype=tf.float32, initializer=w_init())
        self.mu = tf.matmul(encoded, W_mu) + b_mu

        W_sigma = tf.get_variable(name='W_sigma', shape=[sizes[-1], self.L], dtype=tf.float32, initializer=w_init())
        b_sigma = tf.get_variable(name='b_sigma', shape=[self.L], dtype=tf.float32, initializer=w_init())
        log_sigma = tf.matmul(encoded, W_sigma) + b_sigma
        self.sigma = tf.nn.elu(log_sigma) + 1 + 1e-14


    def __build_loss(self):
        hop_pos = tf.stack([self.triplets[:, 0], self.triplets[:, 1]], 1)
        hop_neg = tf.stack([self.triplets[:, 0], self.triplets[:, 2]], 1)
        eng_pos = self.energy_kl(hop_pos)
        eng_neg = self.energy_kl(hop_neg)
        energy = tf.square(eng_pos) + tf.exp(-eng_neg)
        self.loss = tf.reduce_mean(energy)

    def energy_kl(self, pairs):
        """
        Computes the energy of a set of node pairs as the KL divergence between their respective Gaussian embeddings.
        Parameters
        ----------
        pairs : array-like, shape [?, 2]
            The edges/non-edges for which the energy is calculated
        Returns
        -------
        energy : array-like, shape [?]
            The energy of each pair given the currently learned model
        """
        ij_mu = tf.gather(self.mu, pairs)
        ij_sigma = tf.gather(self.sigma, pairs)

        sigma_ratio = ij_sigma[:, 1] / ij_sigma[:, 0]
        trace_fac = tf.reduce_sum(sigma_ratio, 1)
        log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)

        mu_diff_sq = tf.reduce_sum(tf.square(ij_mu[:, 0] - ij_mu[:, 1]) / ij_sigma[:, 0], 1)

        return 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

    def __dataset_generator(self, hops, scale_terms):
        """
        Generates a set of triplets and associated scaling terms by:
            1. Sampling for each node a set of nodes from each of its neighborhoods
            2. Forming all implied pairwise constraints
        Uses tf.Dataset API to perform the sampling in a separate thread for increased speed.
        Parameters
        ----------
        hops : dict
            A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
        scale_terms : dict
            The appropriate up-scaling terms to ensure unbiased estimates for each neighbourhood
        Returns
        -------
        """
        def gen():
            while True:
                yield to_triplets(sample_all_hops(hops), scale_terms)

        dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.float32), ([None, 3], [None]))
        self.triplets, self.scale_terms = dataset.prefetch(1).make_one_shot_iterator().get_next()

    def __save_vars(self, sess):
        """
        Saves all the trainable variables in memory. Used for early stopping.
        Parameters
        ----------
        sess : tf.Session
            Tensorflow session used for training
        """
        self.saved_vars = {var.name: (var, sess.run(var)) for var in tf.trainable_variables()}

    def __restore_vars(self, sess):
        """
        Restores all the trainable variables from memory. Used for early stopping.
        Parameters
        ----------
        sess : tf.Session
            Tensorflow session used for training
        """
        for name in self.saved_vars:
                sess.run(tf.assign(self.saved_vars[name][0], self.saved_vars[name][1]))

    def train(self, gpu_list='0'):
        """
        Trains the model.
        Parameters
        ----------
        gpu_list : string
            A list of available GPU devices.
        Returns
        -------
        sess : tf.Session
            Tensorflow session that can be used to obtain the trained embeddings
        """
        early_stopping_score_max = -float('inf')
        tolerance = self.tolerance

        train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)

        #sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=gpu_list, allow_growth=True)))
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        for epoch in range(self.max_iter):
            loss, _ = sess.run([self.loss, train_op], self.feed_dict)

            early_stopping_score = -loss
            if epoch % 50 == 0:
                logger.debug('epoch: {:3d}, loss: {:.4f}'.format(epoch, loss))

            if early_stopping_score > early_stopping_score_max:
                early_stopping_score_max = early_stopping_score
                tolerance = self.tolerance
                self.__save_vars(sess)
            else:
                tolerance -= 1

            if tolerance == 0:
                break
        
        if tolerance > 0:
            logger.warn('WARNING: Training might not have converged. Try increasing max_iter')
                  
        self.__restore_vars(sess)

        return sess

class Model:
    def __init__(self, embedding_size=256):
        self.embedding_size = embedding_size
        self.model = None

    def fit(self, X, M):
        logger.debug("Building vocab")
        self.vocab = idne.preprocessing.text.dictionary.Dictionary(M, min_df=5, max_df_ratio=0.25)
        logger.debug("Building tf vectors")
        self.tf_vectors = normalize(idne.preprocessing.text.vectorizers.get_tf_dictionary(self.vocab), norm='l1', axis=1)
        logger.debug("Learning g2g")
        self.model = Graph2Gauss(A=X, X=self.tf_vectors, L=self.embedding_size)
        self.sess = self.model.train()
        feed_dict = {self.model.X: sparse_feeder(self.tf_vectors)}
        self.vectors, self.sigma = self.sess.run([self.model.mu, self.model.sigma], feed_dict)
        self.vectors = np.array(self.vectors, dtype=np.float)

    def get_embeddings(self):
        return self.vectors

    def get_embeddings_new(self, M):
        tf_v = idne.preprocessing.text.vectorizers.get_tf_N(self.vocab, M)
        tf_v.data[tf_v.data > 1] = 1
        feed_dict = {self.model.X: sparse_feeder(tf_v)}
        mu, sigma = self.sess.run([self.model.mu, self.model.sigma], feed_dict)
        return np.array(mu, dtype=np.float)

    def predict(self, i, j):
        u = self.get_embeddings()[i]
        v = self.get_embeddings()[j]
        return 1-scipy.spatial.distance.cosine(u,v)

    def predict_new(self, Mi, Mj):
        [u,v] = self.get_embeddings_new([Mi, Mj])
        return 1-scipy.spatial.distance.cosine(u,v)



