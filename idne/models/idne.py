"""
.. module:: idne
   :synopsis: IDNE algorithm
.. moduleauthor:: Robin Brochier <robin.brochier@univ-lyon2.fr>

"""
import logging
logger = logging.getLogger()
import numpy as np
import time
import warnings
import idne.preprocessing.text.dictionary
import idne.preprocessing.text.vectorizers
import idne.models.data_generator
import idne.models.tools
from idne.models.transformer import activation_attention, cosine_loss
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import scipy.spatial.distance
import idne.preprocessing.graph.random_walker
import idne.preprocessing.graph.window_slider
#from fast_pagerank import pagerank_power


from sklearn.preprocessing import normalize

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None

class TFModel():
    """
    The core of the model, defined with tensorflow.
    We declare the full pipeline (constants, variables, placeholders, operators, optimizer)
    """


    def __init__(self, num_words, embedding_size, number_incuding_points):

        self.W = tf.get_variable(
            "W",
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[num_words, embedding_size],
            dtype=tf.float32
        )

        self.inducing_points = tf.get_variable(
            "inducing_points",
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[number_incuding_points, embedding_size],
            dtype=tf.float32
        )

        # Embedding lookups and inputs
        self.input_1_documents = tf.placeholder(tf.int32, shape=[None, None],
                                                name="input_1_documents")  # (batch_size, seq_len)
        self.input_2_documents = tf.placeholder(tf.int32, shape=[None, None],
                                                name="input_2_documents")  # (batch_size, seq_len)
        self.input_x = tf.placeholder(tf.float32, shape=[None], name="input_x")  # (batch_size)

        mask_1 = tf.cast(tf.math.equal(self.input_1_documents, 0), tf.float32)  # (batch_size, seq_len)
        mask_2 = tf.cast(tf.math.equal(self.input_2_documents, 0), tf.float32)  # (batch_size, seq_len)

        w_1 = tf.nn.embedding_lookup(self.W, self.input_1_documents,
                                     name="w_1")  # (batch_size, seq_len, embedding_size)
        w_2 = tf.nn.embedding_lookup(self.W, self.input_2_documents,
                                     name="w_2")  # (batch_size, seq_len, embedding_size)

        activation = lambda x: tf.nn.relu(x)
        #t = 10
        #activation = lambda x: (1/t) * tf.math.log( 1 + tf.exp(t*x))

        attention_1, attention_weights_1, self.alphas_1 = activation_attention(self.inducing_points, w_1, w_1, mask_1, activation)
        attention_2, attention_weights_2, self.alphas_2 = activation_attention(self.inducing_points, w_2, w_2, mask_2, activation)

        #self.covariance_loss = cosine_loss(self.inducing_points)

        self.attention_vectors = [attention_1, attention_2]
        self.attention_weights = [attention_weights_1, attention_weights_2]
        self.alphas = [self.alphas_1, self.alphas_2]

        scalar_product = tf.reduce_sum(tf.multiply(attention_1, attention_2), axis=-1)
        self.score = tf.nn.sigmoid(tf.multiply(self.input_x, scalar_product))

        self.obj_loss = tf.reduce_sum(tf.negative(tf.log(self.score)), name="loss")

        self.loss = self.obj_loss
        self.lr =tf.Variable(0.001)
        #self.optimizer = tf.train.AdamOptimizer(
        # name="optimizer",
        # learning_rate=self.lr
        # ).minimize(self.loss)
        self.optimizer = tf.contrib.opt.LazyAdamOptimizer(
            name="optimizer",
            learning_rate=self.lr
        ).minimize(self.loss)


class Model:

    def __init__(self, embedding_size=256, number_iterations=5e3, number_incuding_points = 32,  number_negative=1, batch_size=16):
        self.is_idne = True
        self.batch_size = batch_size
        self.number_iterations = number_iterations
        self.embedding_size = embedding_size
        self.model = None
        self.number_incuding_points = number_incuding_points
        self.number_negative = number_negative

    def __data_generator(self, X, M, number_negative, number_iterations, batch_size):
        for i, (u, v, x) in enumerate(idne.models.data_generator.async_batches(
                X,
                number_negative,
                number_iterations,
                batch_size
        )):
            # Minibatch Gradient Descent
            try:
                yield [M[k] for k in u], [M[k] for k in v], x
            except StopIteration:
                return


    def fit(self, X, M, vocab = None):

        """
        Fit the model
        :param X: network node similarity matrix
        :param M: document word index sequences
        :param W: word embeddings
        :param number_negative: number of negative exemple to generate for each positive pairs
        :param batch_size: number of positive exemple per batch
        :return: None
        """
        self.vocab = vocab
        logger.debug("Building dictionary")
        if self.vocab is None:
            self.vocab = idne.preprocessing.text.dictionary.Dictionary(M, min_df=5, max_df_ratio=0.25)
        logger.debug("Building sequences")
        seqs = [self.vocab.get_sequence(m) for m in M]
        self.num_words = self.vocab.num_words + 1 # reserves index zero for padding/masking
        self.M = [m+1 for m in seqs]


        self.X = X
        self.X = (self.X + np.dot(self.X, self.X)) / 2


        """
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


        self.X.data[self.X.data<5] = 0
        self.X.eliminate_zeros()
        self.X = normalize(self.X, 'l1', axis=1)
        """

        logger.debug("Shapes of X and M: {0}, {1}".format(X.shape, len(M)))
        num_clocks = int(self.number_iterations)
        self.number_nodes = X.shape[0]
        logger.debug("Total number of iterations: {0}".format(self.number_iterations))
        logger.debug("Total number of samples: {0}".format(self.number_iterations*(self.batch_size*(self.number_negative+1))))
        clock = idne.models.tools.Clock(num_clocks / 100)  #  A clock that will trigger every percent of progression
        self.graph = tf.Graph()  # We create a computational graph (object that will store the tensorflow pipeline)
        with self.graph.as_default():
            self.session = tf.Session()  # We create a session (object that will instantiate the graph)
            with self.session.as_default():
                self.model = TFModel(self.num_words, self.embedding_size, self.number_incuding_points)
                init_op = tf.global_variables_initializer()  # we create an intitialization operation
                self.session.run(init_op)  # Here the tf variable are initialized
                generate_time = 0
                compute_time = 0
                generate_time_ref = time.time()
                k = 0
                loss_average = 0
                loss_average_count = 0
                logger.debug("Building generator")
                half_it = self.number_iterations/2
                for i, (uM, vM, x) in enumerate(self.__data_generator(
                        self.X,
                        self.M,
                        self.number_negative,
                        int(self.number_iterations),
                        self.batch_size
                )):
                    max_len_u = max([len(m) for m in uM])
                    max_len_v = max([len(m) for m in vM])

                    upM = np.zeros((len(uM), max_len_u))
                    vpM = np.zeros((len(vM), max_len_v))
                    for l, m in enumerate(uM):
                        upM[l, 0:len(m)] = m
                    for l, m in enumerate(vM):
                        vpM[l, 0:len(m)] = m

                    generate_time += (time.time() - generate_time_ref)
                    feed_dict = {
                        self.model.input_1_documents: upM,
                        self.model.input_2_documents: vpM,
                        self.model.input_x: x,
                    }
                    compute_time_ref = time.time()

                    _, loss, obj_loss, score, lr = self.session.run([self.model.optimizer,
                                                                     self.model.loss, self.model.obj_loss,
                                                                     self.model.score, self.model.lr],
                                                                    feed_dict=feed_dict)
                    k += 1
                    N = x.shape[0]
                    loss_average += obj_loss
                    loss_average_count += 1
                    compute_time += (time.time() - compute_time_ref)
                    if clock.update():
                        T = self.number_iterations
                        k = 0.1
                        i_lr = 0.001
                        #lr_assign_op = self.model.lr.assign(i_lr * (T - i) / T)
                        #lr_assign_op = self.model.lr.assign(i_lr * np.exp(i/T))
                        #self.session.run(lr_assign_op)
                        ratio = i / num_clocks
                        tot_time = generate_time + compute_time
                        logger.debug(
                            "{0:.2f}% - loss={1:.3f} - lossAv({7})={6:.3f} - lossObj={2:.3f} - score={3:.5f} - gt={4:.2f}% - ct={5:.2f} - lr={8:.5f}".format(
                                ratio * 100,
                                loss / N,
                                obj_loss / N,
                                score.mean(),
                                generate_time * 100 / tot_time,
                                compute_time * 100 / tot_time,
                                loss_average / (N * loss_average_count),
                                loss_average_count * (self.batch_size * (self.number_negative + 1)),
                                lr
                            ))
                        k = 0
                        loss_average = 0
                        loss_average_count = 0
                    generate_time_ref = time.time()
                self.vectors = np.zeros((len(self.M), self.embedding_size))
                for i, m in enumerate(self.M):
                    feed_dict = {
                        self.model.input_1_documents: [m],
                        self.model.input_2_documents: [m],
                        self.model.input_x: [1],
                    }
                    vecs, = self.session.run([self.model.attention_vectors], feed_dict=feed_dict)
                    self.vectors[i] = vecs[0][0]

    def plot_topics(self):
        words = self.vocab.ids_to_words
        PP = self.get_inducing_points()
        w_counts = np.zeros((PP.shape[0], self.num_words))
        w = np.zeros((PP.shape[0],  self.num_words))
        a = np.zeros((PP.shape[0],len(self.M)))
        for i,m in enumerate(self.M):
            w1, w2, a1, a2 = self.get_weights_and_alphas(m,m)
            w[:,m] += w1 * a1[:,np.newaxis]
            a[:,i] = a1
            w_counts[:,m] += 1
        w /= w_counts
        a_std = a.std(axis=1)
        sims = w[:, 1:]
        alphas = a.mean(axis=1)
        sorting_topics = np.argsort(alphas)[::-1]
        closest = np.argsort(sims, axis=1)[:, ::-1]
        print("\n\n plot_topics")
        for i in sorting_topics:
            print(f"\n Topic {i}: alpha={alphas[i] * 100:0.3f}% +- {a_std[i] * 100:0.3f}%")
            plt.plot(sims[i][closest[i]])
            for j in closest[i][:10]:
                print(f"{sims[i, j]:0.3f} => {words[j]} ")
        plt.show()
        for i in sorting_topics:
            plt.plot(np.sort(a[i])[::-1])
        plt.show()

    def plot_words_topics_amplitudes(self):
        T = self.get_inducing_points()
        W = self.get_word_embeddings()
        T_n = np.linalg.norm(T, axis=1)
        W_n = np.linalg.norm(W, axis=1)
        T_i = np.argsort(T_n)[::-1]
        W_i = np.argsort(W_n)[::-1]

        words = self.vocab.ids_to_words
        plt.plot(T_n[T_i], label="topics")
        plt.show()
        plt.plot(W_n[W_i], label="words")
        plt.show()

        print("\n\n plot_words_topics_amplitudes")

        print("Largest word norms:")
        for i in W_i[:20]:
            print(words[i], "=", W_n[i])
        print()
        print("Smallest word norms:")
        for i in W_i[-20:]:
            print(words[i], "=", W_n[i])



    def plot_direct_topics(self):
        print("\n\n plot_direct_topics")
        words = self.vocab.ids_to_words
        W = self.get_word_embeddings()
        PP = self.get_inducing_points()
        sims = PP.dot(W.T)
        sims = np.maximum(0,PP.dot(W.T))
        #sims = normalize(sims, 'l1', axis=0)

        alphas = sims.sum(axis=1)
        alphas = alphas / alphas.sum()
        sorting_topics = np.argsort(alphas)[::-1]
        closest = np.argsort(sims, axis=1)[:, ::-1]
        for i in sorting_topics:
            #for j in closest[i][:20]:
            #    print(f"{sims[i, j]:0.3f} => {words[j]} ")
            print(f"\n Topic {i+1} ({alphas[i] * 100:0.1f}%) &{', '.join([words[j] for j in closest[i][:20]])}")

        #plt.show()

    def raw_attention(self, text):
        d = self.vocab.get_sequence(text)
        words = self.vocab.ids_to_words
        text_processed = [words[i] for i in d]
        W = self.get_word_embeddings()
        Wd = W[d]
        PP = self.get_inducing_points()
        aw = PP.dot(Wd.T)
        if aw.shape[1] == 0:
            aw = np.zeros((PP.shape[0], 1))
        aw = np.maximum(0, aw)
        aw = normalize(aw, 'l1', axis=0)
        return aw, text_processed

    def get_topics(self):
        max_len = max([len(m) for m in self.M])
        M = np.zeros((len(self.M), max_len))
        for l, m in enumerate(self.M):
            M[l, 0:len(m)] = m
        with self.graph.as_default():
            with self.session.as_default():
                feed_dict = {
                    self.model.input_1_documents: M
                }
                alphas = self.session.run([self.model.alphas_1], feed_dict=feed_dict)[0]
                return alphas

    def get_weights_and_alphas(self, Mi, Mj):
        with self.graph.as_default():
            with self.session.as_default():
                feed_dict = {
                    self.model.input_1_documents: [Mi],
                    self.model.input_2_documents: [Mj],
                    self.model.input_x: [1],
                }
                attention_weights, alphas = self.session.run([self.model.attention_weights, self.model.alphas], feed_dict=feed_dict)
                w1, w2 = attention_weights[0][0], attention_weights[1][0]
                a1, a2 = alphas[0][0], alphas[1][0]
                return w1, w2, a1, a2

    def get_inducing_points(self):
        with self.graph.as_default():
            with self.session.as_default():
                feed_dict = {}
                inducing_points, = self.session.run([self.model.inducing_points], feed_dict=feed_dict)
                return inducing_points

    def get_word_embeddings(self):
        with self.graph.as_default():
            with self.session.as_default():
                feed_dict = {}
                W, = self.session.run([self.model.W], feed_dict=feed_dict)
                return W[1:]

    def get_embeddings(self):
       return self.vectors

    def get_embeddings_new(self, docs):
        M = list()
        for i, d in enumerate(docs):
            M.append(self.vocab.get_sequence(d))
        M = [m + 1 for m in M]
        attention_vectors = np.zeros((len(M), self.embedding_size))
        with self.graph.as_default():
            with self.session.as_default():
                for i, m in enumerate(M):
                    feed_dict = {
                        self.model.input_1_documents: [m],
                        self.model.input_2_documents: [m],
                        self.model.input_x: [1],
                    }
                    vecs, = self.session.run([self.model.attention_vectors], feed_dict=feed_dict)
                    attention_vectors[i] = vecs[0][0]
        return attention_vectors

    def predict(self, i, j):
        u = self.get_embeddings()[i]
        v = self.get_embeddings()[j]
        return 1-scipy.spatial.distance.cosine(u, v)

    def predict_new(self, Mi, Mj):
        [u, v] = self.get_embeddings_new([Mi, Mj])
        return 1-scipy.spatial.distance.cosine(u, v)




