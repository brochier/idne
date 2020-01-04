import os
import numpy as np
import tensorflow as tf
import time
import math
import logging
import warnings
warnings.filterwarnings('ignore')
import idne.preprocessing.text.dictionary
import idne.preprocessing.text.vectorizers
logger = logging.getLogger()
import idne.preprocessing.graph.random_walker
import idne.preprocessing.graph.window_slider
import scipy.spatial.distance


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Clock():
    def __init__(self, epochs):
        self.clock = 0
        self.clocks = dict()
        self.limits = dict()
        self.status = dict()
        self.epochs = epochs
        for epoch in self.epochs:
            self.clocks[epoch] = 0
            self.limits[epoch] = epoch
            self.status[epoch] = True

    def update(self):
        for epoch in self.epochs:
            self.clocks[epoch] += 1
            if self.limits[epoch] == self.clocks[epoch]:
                self.status[epoch] = True
                self.clocks[epoch] = 0

    def check(self, epoch):
        if self.status[epoch] == True:
            self.status[epoch] = False
            return True

class TFModel(object):
    def __init__(self,
                 embedding_size, # embedding dim for both input/output
                 learn_rate, # initial learning rate for gradient descent
                 j_vector_size, # dim of output vectors
                 i_index_size, # number of nodes in input space
                 pretrained_word_embeddings=None
                ):

        self.embedding_size = embedding_size
        self.init_range = 1/embedding_size
        self.learn_rate = learn_rate
        self.j_vector_size = j_vector_size
        self.i_index_size = i_index_size
        self.indexJ = None
        self.vectorJ = None
        self.indexI = None
        self.wi = None
        self.wj = None
        self.bi = None
        self.bj = None

        self.indexI = tf.placeholder(tf.int32, shape=[None], name="indexI")
        self.IW = tf.Variable(
            tf.random_uniform([self.i_index_size, self.embedding_size], -self.init_range, self.init_range), name="IW")
        self.IB = tf.Variable(tf.random_uniform([self.i_index_size], -self.init_range, self.init_range), name="IB")
        self.wi = tf.nn.embedding_lookup(self.IW, self.indexI, name="wi")
        self.bi = tf.nn.embedding_lookup(self.IB, self.indexI, name="bi")

        self.vectorJ = tf.sparse_placeholder(tf.float32, name="inputJ")
        if pretrained_word_embeddings is None:
            self.JW = tf.Variable(
                tf.random_uniform([self.j_vector_size, self.embedding_size], -self.init_range, self.init_range), name="JW")
        else:
            self.JW = tf.Variable(pretrained_word_embeddings,
                                 name="JW",
                                 trainable=True)


        lookup = tf.sparse_tensor_dense_matmul(self.vectorJ, self.JW, adjoint_a=False, adjoint_b=False, name="wj")
        sums = tf.sparse_reduce_sum(self.vectorJ, 1, keep_dims=True)
        self.wj = tf.divide(lookup, sums)

        self.Xij = tf.placeholder(tf.float32, shape=[None], name="Xij")

        wiwjProduct = tf.reduce_sum(tf.multiply(self.wi,self.wj), 1)
        logXij = tf.log(1+self.Xij)

        dist = tf.square(tf.add_n([wiwjProduct, self.bi, tf.negative(logXij)]))

        self.loss = tf.reduce_sum(dist, name="loss")

        tf.summary.scalar("loss", self.loss)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learnRate = tf.Variable(learn_rate, trainable=False, name="learnRate")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learnRate, name="optimizer").minimize(
                self.loss, global_step=self.global_step)




class Model:

    def __init__(self, saver = None, callback = None):
        self.saver = saver
        self.callback = callback
        self.X = None # nodes cooccurrences matrix
        self.M = None  # nodes feature matrix
        self.J_matrix = None
        self.I_matrix = None
        self.J_vectors = None
        self.I_vectors = None

    def chuncker(self, iterable, n):
        """
        grouper([ABCDEFG], 3) --> [[ABC],[DEF],[G]]
        """
        ind = range(0, len(iterable), n)
        for i in range(len(ind) - 1):
            try:
                yield iterable[ind[i]:ind[i + 1]]
            except StopIteration:
                return
        if ind[-1] < len(iterable):
            try:
                yield iterable[ind[-1]:len(iterable)]
            except StopIteration:
                return

    def generate_batches(self):
        logger.debug("Removing empty text nodes...")
        mask = set(np.where(np.squeeze(np.asarray(self.M.sum(axis=1) == 0)))[0])
        self.forbidden_neg = np.array(list(mask), dtype=np.int)
        for k, (i, j) in enumerate(zip(self.X.nonzero()[0], self.X.nonzero()[1])):
            if i in mask or j in mask:
                self.X.data[k] = 0
        self.X.eliminate_zeros()

        sparse_ind = [s.nonzero()[1] for s in self.M]
        sparse_vec = [s.data for s in self.M]
        sparse_indices = sparse_ind
        sparse_vectors = sparse_vec
        M_dim = self.M.shape[1]

        """
        occurrences_counts = self.X.sum(axis=1)
        occurrences_counts[occurrences_counts == 0] = 1
        flatten_probs = np.power(1 / occurrences_counts, 0.75)
        nodes_negsampling_prob = np.squeeze(np.asarray(flatten_probs / flatten_probs.sum()))
        """

        data = np.array(self.X.data, dtype=np.float32)
        indices = self.X.nonzero()
        cols = np.array(indices[1], dtype=np.int32)
        rows = np.array(indices[0], dtype=np.int32)

        M = len(cols)
        cols_neg = np.tile(cols, self.k_neg)
        rows_neg = np.random.randint(0, self.X.shape[0], self.k_neg*M)
        #rows_neg = np.random.choice(np.arange(self.X.shape[0]), size=self.k_neg*M, p=nodes_negsampling_prob)

        cols = np.hstack((cols,cols_neg))
        rows = np.hstack((rows, rows_neg))
        data = np.hstack((data, np.zeros(self.k_neg*M)))

        ind = np.arange(len(data))
        np.random.shuffle(ind)
        data = data[ind]
        cols = cols[ind]
        rows = rows[ind]
        logger.debug("Shape of X=%s", self.X.shape)
        for ind in self.chuncker(range(0, len(data)), self.batch_size):
            N = len(ind)
            values = np.array([v for i,k in enumerate(cols[ind]) for v in sparse_vectors[k]], dtype=np.float32)
            indices = np.array([[l, v] for l, k in enumerate(cols[ind]) for m, v in enumerate(sparse_indices[k])], dtype=np.int64)
            dense_shape = np.array([N, M_dim], dtype=np.float32)
            Jvectors = tf.SparseTensorValue(indices, values, dense_shape)
            yield rows[ind], Jvectors, data[ind]


    def fit(self,
            X,
            M,
            pretrained_word_embeddings = None,
            embedding_size = 256,
            batch_size = 128,
            n_epochs = 1,
            learn_rate=0.001,
            k_neg = 1,
            x_min = 5
            ):

        random_walker = idne.preprocessing.graph.random_walker.RandomWalker(
            X,
            walks_length=40,
            walks_number=10
        )
        random_walks = random_walker.build_random_walks()
        slider = idne.preprocessing.graph.window_slider.WindowSlider(
            random_walks,
            X.shape[0],
            window_size=5,
            window_factor="decreasing"
        )
        self.X = slider.build_cooccurrence_matrix()
        self.x_min = x_min
        logger.debug("Size of X.data before filtering:%s", len(self.X.data))
        self.X.data[self.X.data <= self.x_min] = 0
        self.X.eliminate_zeros()
        logger.debug("Size of X.data after filtering:%s", len(self.X.data))
        logger.debug(" ".join([str(v) for v in ["Shape of cooccurrences matrix : ", X.shape]]))
        logger.debug(" ".join([str(v) for v in ["Density of cooccurrences matrix: ", " (", len(X.data) * 100 / (X.shape[0] * X.shape[1]), "%)"]]))

        logger.debug(
            "X => Min=%s/Max=%s/Mean=%s/Std=%s/Sum=%s:",
            X.data.min(),
            X.data.max(),
            X.data.mean(),
            X.data.std(),
            X.data.sum()
        )
        self.vocab = idne.preprocessing.text.dictionary.Dictionary(M)
        logger.debug("Building tf vectors")
        self.M = idne.preprocessing.text.vectorizers.get_tf_dictionary(self.vocab)
        logger.debug(" ".join([str(v) for v in ["Shape of features matrix : ", self.M.shape]]))

        self.links_count = len(self.X.data)
        self.i_vector_size = self.M.shape[1]
        self.j_vector_size = self.M.shape[1]
        self.i_index_size = self.X.shape[0]
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.embedding_size = embedding_size  # dimension of embeddings
        self.k_neg = k_neg
        self.num_batches = math.ceil(self.links_count*(1+self.k_neg) / self.batch_size) * n_epochs
        self.learn_rate = learn_rate

        logger.debug("Number of data={0}, number of batches={1}, number of epochs={2}".format(self.links_count, self.num_batches, n_epochs))
        loss = 0
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                model = TFModel(
                      embedding_size = self.embedding_size,
                      learn_rate = self.learn_rate,
                      j_vector_size = self.j_vector_size,
                      i_index_size = self.i_index_size,
                      pretrained_word_embeddings = pretrained_word_embeddings,
                )
                lr = 0
                init_op = tf.global_variables_initializer()
                self.session.run(init_op)
                start = time.time()
                clock = Clock([10, 100, 1000, 10000])
                saver_step = 0.1
                epoch_count = 0
                for epoch in np.arange(self.n_epochs):
                    for i, (Iindices, Jvectors, Xij) in enumerate(self.generate_batches()):
                        clock.update()
                        feed_dict = {
                            model.indexI: Iindices,
                            model.vectorJ: Jvectors,
                            model.Xij: Xij
                        }
                        _, loss, lr, gs = self.session.run([model.optimizer,
                                                       model.loss,
                                                       model.learnRate,
                                                       model.global_step],
                                                      feed_dict=feed_dict)
                        loss = loss / self.batch_size
                        progression = gs / self.num_batches
                        if gs % (self.num_batches // 1000) == 0:
                            now = time.strftime("%H:%M:%S")
                            logger.debug("{} Progression={:1} Ep={:2}  GS={:5.2e}  LR={:5.2e}  Loss={:4.3f}  Speed={:5.2e}s/sec".format(
                                now, int(progression*100), epoch, gs, lr, loss,  self.batch_size * gs / (time.time() - start)))
                        if self.saver is not None and progression >= saver_step:
                            saver_step += 0.1
                            emb_types = ["IJ", "I", "J"]
                            for e in emb_types:
                                self.I_matrix = self.graph.get_tensor_by_name("IW:0").eval()
                                self.J_matrix = self.graph.get_tensor_by_name("JW:0").eval()
                                self.I_vectors = self.I_matrix
                                self.J_vectors = self.sparse_embeddings()
                                self.saver(self.get_embeddings(e), e + "_" + str(gs % (self.num_batches // 10)))
                        if self.callback is not None and epoch == epoch_count:
                            epoch_count += 1
                            self.I_matrix = self.graph.get_tensor_by_name("IW:0").eval()
                            self.J_matrix = self.graph.get_tensor_by_name("JW:0").eval()
                            self.I_vectors = self.I_matrix
                            self.J_vectors = self.sparse_embeddings()
                            embI = self.get_embeddings()
                            embJ = self.get_embeddings()
                            embIJ = self.get_embeddings()
                            self.callback(embI, embJ, embIJ, epoch, loss)

        with self.session.as_default():
            self.I_matrix = self.graph.get_tensor_by_name("IW:0").eval()
            self.J_matrix = self.graph.get_tensor_by_name("JW:0").eval()
            self.I_vectors = self.I_matrix
            self.J_vectors = self.sparse_embeddings()
        if self.saver is not None:
            emb_types = ["IJ", "I", "J"]
            for e in emb_types:
                self.saver(self.get_embeddings(e), e)
        if self.callback is not None:
            embI = self.get_embeddings()
            embJ = self.get_embeddings()
            embIJ = self.get_embeddings()
            self.callback(embI, embJ, embIJ, self.n_epochs, loss)

        self.vectors = np.hstack([self.I_vectors, self.J_vectors])

    def sparse_embeddings(self):
        sums = self.M.sum(axis=1)
        sums[sums==0] = 1
        return self.M.dot(self.J_matrix) / sums

    def get_embeddings(self):
        return self.vectors

    def get_embeddings_new(self, docs):
        M = idne.preprocessing.text.vectorizers.get_tf_N(self.vocab, docs)
        sums = M.sum(axis=1)
        sums[sums == 0] = 1
        return M.dot(self.J_matrix) / sums

    def predict(self, i, j):
        u = self.get_embeddings()[i]
        v = self.get_embeddings()[j]
        return 1-scipy.spatial.distance.cosine(u, v)

    def predict_new(self, Mi, Mj):
        [u, v] = self.get_embeddings_new([Mi, Mj])
        return 1-scipy.spatial.distance.cosine(u, v)




