import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import random

MAX_LEN = 300
neg_table_size = 1000000
NEG_SAMPLE_POWER = 0.75
batch_size = 64
num_epoch = 200
embed_size = 200
lr = 1e-3

rho1 = 1
rho2 = 0.3
rho3 = 0.3

import logging
logger = logging.getLogger()

class TFModel:
    def __init__(self, vocab_size, num_nodes):
        self.rho1 = float(rho1)
        self.rho2 = float(rho2)
        self.rho3 = float(rho3)
        # '''hyperparameter'''
        with tf.name_scope('read_inputs') as scope:
            self.Text_a = tf.placeholder(tf.int32, [batch_size, MAX_LEN], name='Ta')
            self.Text_b = tf.placeholder(tf.int32, [batch_size, MAX_LEN], name='Tb')
            self.Text_neg = tf.placeholder(tf.int32, [batch_size, MAX_LEN], name='Tneg')
            self.Node_a = tf.placeholder(tf.int32, [batch_size], name='n1')
            self.Node_b = tf.placeholder(tf.int32, [batch_size], name='n2')
            self.Node_neg = tf.placeholder(tf.int32, [batch_size], name='n3')

        with tf.name_scope('initialize_embedding') as scope:
            self.text_embed = tf.Variable(tf.truncated_normal([vocab_size, embed_size // 2], stddev=0.3))
            self.node_embed = tf.Variable(tf.truncated_normal([num_nodes, embed_size // 2], stddev=0.3))
            self.node_embed = tf.clip_by_norm(self.node_embed, clip_norm=1, axes=1)

        with tf.name_scope('lookup_embeddings') as scope:
            self.TA = tf.nn.embedding_lookup(self.text_embed, self.Text_a)
            self.T_A = tf.expand_dims(self.TA, -1)

            self.TB = tf.nn.embedding_lookup(self.text_embed, self.Text_b)
            self.T_B = tf.expand_dims(self.TB, -1)

            self.TNEG = tf.nn.embedding_lookup(self.text_embed, self.Text_neg)
            self.T_NEG = tf.expand_dims(self.TNEG, -1)

            self.N_A = tf.nn.embedding_lookup(self.node_embed, self.Node_a)
            self.N_B = tf.nn.embedding_lookup(self.node_embed, self.Node_b)
            self.N_NEG = tf.nn.embedding_lookup(self.node_embed, self.Node_neg)
        self.convA, self.convB, self.convNeg = self.conv()
        self.loss = self.compute_loss()

    def conv(self):
        W2 = tf.Variable(tf.truncated_normal([2, embed_size // 2, 1, 100], stddev=0.3))
        rand_matrix = tf.Variable(tf.truncated_normal([100, 100], stddev=0.3))

        convA = tf.nn.conv2d(self.T_A, W2, strides=[1, 1, 1, 1], padding='VALID')
        convB = tf.nn.conv2d(self.T_B, W2, strides=[1, 1, 1, 1], padding='VALID')
        convNEG = tf.nn.conv2d(self.T_NEG, W2, strides=[1, 1, 1, 1], padding='VALID')

        hA = tf.tanh(tf.squeeze(convA))
        hB = tf.tanh(tf.squeeze(convB))
        hNEG = tf.tanh(tf.squeeze(convNEG))

        tmphA = tf.reshape(hA, [batch_size * (MAX_LEN - 1), embed_size // 2])
        ha_mul_rand = tf.reshape(tf.matmul(tmphA, rand_matrix),
                                 [batch_size, MAX_LEN - 1, embed_size // 2])
        r1 = tf.matmul(ha_mul_rand, hB, adjoint_b=True)
        r3 = tf.matmul(ha_mul_rand, hNEG, adjoint_b=True)
        att1 = tf.expand_dims(tf.stack(r1), -1)
        att3 = tf.expand_dims(tf.stack(r3), -1)

        att1 = tf.tanh(att1)
        att3 = tf.tanh(att3)

        pooled_A = tf.reduce_mean(att1, 2)
        pooled_B = tf.reduce_mean(att1, 1)
        pooled_NEG = tf.reduce_mean(att3, 1)

        a_flat = tf.squeeze(pooled_A)
        b_flat = tf.squeeze(pooled_B)
        neg_flat = tf.squeeze(pooled_NEG)

        w_A = tf.nn.softmax(a_flat)
        w_B = tf.nn.softmax(b_flat)
        w_NEG = tf.nn.softmax(neg_flat)

        rep_A = tf.expand_dims(w_A, -1)
        rep_B = tf.expand_dims(w_B, -1)
        rep_NEG = tf.expand_dims(w_NEG, -1)

        hA = tf.transpose(hA, perm=[0, 2, 1])
        hB = tf.transpose(hB, perm=[0, 2, 1])
        hNEG = tf.transpose(hNEG, perm=[0, 2, 1])

        rep1 = tf.matmul(hA, rep_A)
        rep2 = tf.matmul(hB, rep_B)
        rep3 = tf.matmul(hNEG, rep_NEG)

        attA = tf.squeeze(rep1)
        attB = tf.squeeze(rep2)
        attNEG = tf.squeeze(rep3)

        return attA, attB, attNEG

    def compute_loss(self):
        p1 = tf.reduce_sum(tf.multiply(self.convA, self.convB), 1)
        p1 = tf.log(tf.sigmoid(p1) + 0.001)

        p2 = tf.reduce_sum(tf.multiply(self.convA, self.convNeg), 1)
        p2 = tf.log(tf.sigmoid(-p2) + 0.001)

        p3 = tf.reduce_sum(tf.multiply(self.N_A, self.N_B), 1)
        p3 = tf.log(tf.sigmoid(p3) + 0.001)

        p4 = tf.reduce_sum(tf.multiply(self.N_A, self.N_NEG), 1)
        p4 = tf.log(tf.sigmoid(-p4) + 0.001)

        p5 = tf.reduce_sum(tf.multiply(self.convB, self.N_A), 1)
        p5 = tf.log(tf.sigmoid(p5) + 0.001)

        p6 = tf.reduce_sum(tf.multiply(self.convNeg, self.N_A), 1)
        p6 = tf.log(tf.sigmoid(-p6) + 0.001)

        p7 = tf.reduce_sum(tf.multiply(self.N_B, self.convA), 1)
        p7 = tf.log(tf.sigmoid(p7) + 0.001)

        p8 = tf.reduce_sum(tf.multiply(self.N_B, self.convNeg), 1)
        p8 = tf.log(tf.sigmoid(-p8) + 0.001)

        rho1 = self.rho1
        rho2 = self.rho2
        rho3 = self.rho3
        temp_loss = rho1 * (p1 + p2) + rho2 * (p3 + p4) + rho3 * (p5 + p6) + rho3 * (p7 + p8)
        loss = -tf.reduce_sum(temp_loss)
        return loss

class dataSet:
    def __init__(self, M, X):
        self.M = M
        self.X = X
        self.edges = self.load_edges(self.X)
        self.text, self.num_vocab, self.num_nodes = self.load_text(self.M)
        self.negative_table = self.InitNegTable(self.edges)


    def load_edges(self, graph):
        r, c = graph.nonzero()
        edges = list(zip(r,c))
        logger.debug(f"Total load {len(edges)} edges.")
        return edges

    def load_text(self, texts):
        vocab = learn.preprocessing.VocabularyProcessor(MAX_LEN)
        text = np.array(list(vocab.fit_transform(texts)), dtype=np.int)
        num_vocab = len(vocab.vocabulary_)
        num_nodes = text.shape[0]
        return text, num_vocab, num_nodes

    def InitNegTable(self, edges):
        a_list, b_list = zip(*edges)
        a_list = list(a_list)
        b_list = list(b_list)
        node = a_list
        node.extend(b_list)
        node_degree = {}
        for i in node:
            if i in node_degree:
                node_degree[i] += 1
            else:
                node_degree[i] = 1
        sum_degree = 0
        for i in node_degree.values():
            sum_degree += pow(i, 0.75)
        por = 0
        cur_sum = 0
        vid = -1
        neg_table = []
        degree_list = list(node_degree.values())
        node_id = list(node_degree.keys())
        for i in range(neg_table_size):
            if ((i + 1) / float(neg_table_size)) > por:
                cur_sum += pow(degree_list[vid + 1], NEG_SAMPLE_POWER)
                por = cur_sum / sum_degree
                vid += 1
            neg_table.append(node_id[vid])
        return neg_table

    def negative_sample(self, edges):
        node1, node2 = zip(*edges)
        sample_edges = []
        func = lambda: self.negative_table[random.randint(0, neg_table_size - 1)]
        for i in range(len(edges)):
            neg_node = func()
            while node1[i] == neg_node or node2[i] == neg_node:
                neg_node = func()
            sample_edges.append([node1[i], node2[i], neg_node])

        return sample_edges

    def generate_batches(self, mode=None):
        edges = self.edges
        if mode != 'add':
            random.shuffle(edges)
            edges = edges[:10000]  # set max number of edges to scale
        num_batch = len(edges) // batch_size
        sample_edges = edges[:num_batch * batch_size]
        sample_edges = self.negative_sample(sample_edges)
        batches = []
        for i in range(num_batch):
            batches.append(sample_edges[i * batch_size:(i + 1) * batch_size])
        return batches

class Model:
    def __init__(self):
        pass

    def fit(self, X, M):
        self.X = X
        self.M = M
        data = dataSet(M, X)
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                model = TFModel(data.num_vocab, data.num_nodes)
                opt = tf.train.AdamOptimizer(lr)
                #opt = tf.contrib.opt.LazyAdamOptimizer(lr)
                train_op = opt.minimize(model.loss)
                sess.run(tf.global_variables_initializer())

                # training
                logger.debug('start training...')
                for epoch in range(num_epoch):
                    loss_epoch = 0
                    batches = data.generate_batches()
                    h1 = 0
                    num_batch = len(batches)
                    for i in range(num_batch):
                        batch = batches[i]

                        node1, node2, node3 = zip(*batch)
                        node1, node2, node3 = np.array(node1), np.array(node2), np.array(node3)
                        text1, text2, text3 = data.text[node1], data.text[node2], data.text[node3]

                        feed_dict = {
                            model.Text_a: text1,
                            model.Text_b: text2,
                            model.Text_neg: text3,
                            model.Node_a: node1,
                            model.Node_b: node2,
                            model.Node_neg: node3
                        }
                        # run the graph
                        _, loss_batch = sess.run([train_op, model.loss], feed_dict=feed_dict)

                        loss_epoch += loss_batch
                    logger.debug(f'epoch: {epoch + 1},  loss: {loss_epoch}')

                # Get embeddings
                batches = data.generate_batches(mode='add')
                num_batch = len(batches)
                embed = [[] for _ in range(data.num_nodes)]
                for i in range(num_batch):
                    batch = batches[i]
                    node1, node2, node3 = zip(*batch)
                    node1, node2, node3 = np.array(node1), np.array(node2), np.array(node3)
                    text1, text2, text3 = data.text[node1], data.text[node2], data.text[node3]

                    feed_dict = {
                        model.Text_a: text1,
                        model.Text_b: text2,
                        model.Text_neg: text3,
                        model.Node_a: node1,
                        model.Node_b: node2,
                        model.Node_neg: node3
                    }

                    # run the graph
                    convA, convB, TA, TB = sess.run([model.convA, model.convB, model.N_A, model.N_B],
                                                    feed_dict=feed_dict)
                    for i in range(batch_size):
                        em = list(np.hstack((TA[i], convA[i])))
                        embed[node1[i]].append(em)
                        em = list(np.hstack((TB[i], convB[i])))
                        embed[node2[i]].append(em)
                self.embeddings = np.zeros((self.X.shape[0], embed_size))
                for i in range(data.num_nodes):
                    if embed[i]:
                        tmp = np.sum(embed[i], axis=0) / len(embed[i])
                        self.embeddings[i] = tmp


    def get_embeddings(self):
        return self.embeddings