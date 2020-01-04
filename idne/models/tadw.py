from sklearn.preprocessing import normalize
import idne.preprocessing.text.dictionary
import idne.preprocessing.text.vectorizers
from sklearn.decomposition import TruncatedSVD
import numpy as np
import scipy.sparse.linalg
import logging
import scipy.spatial.distance

logger = logging.getLogger()


class TADW(object):

    def __init__(self, dim=256, lamb=0.2):
        self.lamb = lamb
        self.dim = int(dim/2)
        self.embeddings = None
        self.I = None
        self.J = None
        self.adj = None
        self.M = None
        self.T = None
        self.node_size = None
        self.feature_size = None

    def fit(self, adjacency_matrix, features, num_iterations=20):
        logger.debug("Building vocab")
        self.vocab = idne.preprocessing.text.dictionary.Dictionary(features, min_df=5, max_df_ratio=0.25)
        logger.debug("Building tfidf vectors")
        tfidf_vectors = idne.preprocessing.text.vectorizers.get_tfidf_dictionary(self.vocab)
        logger.debug("Building svd vectors")

        #Ud, Sd, VT = scipy.sparse.linalg.svds(tfidf_vectors, k=self.dim)
        #svd_vectors = np.array(Ud) * Sd.reshape(self.dim)

        self.svd = TruncatedSVD(n_components=self.dim)
        svd_vectors = self.svd.fit_transform(tfidf_vectors)
        #self.dim = self.svd.components_.shape[0]

        self.adj = adjacency_matrix.A
        self.adj = self.adj / self.adj.sum(axis=1)
        # M=(A+A^2)/2 where A is the row-normalized adjacency matrix
        self.M = (self.adj + np.dot(self.adj, self.adj)) / 2
        # T is feature_size*node_num, text features
        self.T = svd_vectors.T
        self.node_size = self.adj.shape[0]
        self.feature_size = self.T.shape[0]
        self.W = np.random.randn(self.dim, self.node_size)
        self.H = np.random.randn(self.dim, self.feature_size)

        # Update
        for i in range(num_iterations):
            logger.debug(f'Iteration {i}')
            # Update W
            B = np.dot(self.H, self.T)
            drv = 2 * np.dot(np.dot(B, B.T), self.W) - \
                  2 * np.dot(B, self.M.T) + self.lamb * self.W
            Hess = 2 * np.dot(B, B.T) + self.lamb * np.eye(self.dim)
            drv = np.reshape(drv, [self.dim * self.node_size, 1])
            rt = -drv
            dt = rt
            vecW = np.reshape(self.W, [self.dim * self.node_size, 1])
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.node_size))
                Hdt = np.reshape(np.dot(Hess, dtS), [
                    self.dim * self.node_size, 1])

                at = np.dot(rt.T, rt) / np.dot(dt.T, Hdt)
                vecW = vecW + at * dt
                rtmp = rt
                rt = rt - at * Hdt
                bt = np.dot(rt.T, rt) / np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.W = np.reshape(vecW, (self.dim, self.node_size))

            # Update H
            drv = np.dot((np.dot(np.dot(np.dot(self.W, self.W.T), self.H), self.T)
                          - np.dot(self.W, self.M.T)), self.T.T) + self.lamb * self.H
            drv = np.reshape(drv, (self.dim * self.feature_size, 1))
            rt = -drv
            dt = rt
            vecH = np.reshape(self.H, (self.dim * self.feature_size, 1))
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.feature_size))
                Hdt = np.reshape(np.dot(np.dot(np.dot(self.W, self.W.T), dtS), np.dot(self.T, self.T.T))
                                 + self.lamb * dtS, (self.dim * self.feature_size, 1))
                at = np.dot(rt.T, rt) / np.dot(dt.T, Hdt)
                vecH = vecH + at * dt
                rtmp = rt
                rt = rt - at * Hdt
                bt = np.dot(rt.T, rt) / np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.H = np.reshape(vecH, (self.dim, self.feature_size))

        self.vectors = np.hstack(
            (normalize(self.W.T), normalize(np.dot(self.T.T, self.H.T))))

    def get_embeddings(self):
        return self.vectors

    def get_embeddings_new(self, M):
        tfidf = idne.preprocessing.text.vectorizers.get_tfidf_N(self.vocab, M)
        vectors = self.svd.transform(tfidf)
        return normalize(np.dot(vectors, self.H.T))

    def predict(self, i, j):
        vi = self.vectors[i]
        vj = self.vectors[j]
        return float(vi.dot(vj.T))

    def predict_new(self, Mi, Mj):
        tfidf_i = idne.preprocessing.text.vectorizers.get_tfidf_1(self.vocab, Mi)
        tfidf_j = idne.preprocessing.text.vectorizers.get_tfidf_1(self.vocab, Mj)
        svd_i = self.svd.transform(tfidf_i)
        svd_j = self.svd.transform(tfidf_j)
        vi = normalize(np.dot(svd_i.reshape(1, -1), self.H.T))
        vj = normalize(np.dot(svd_j.reshape(1, -1), self.H.T))
        return vi.dot(vj.T)[0][0]


class Model:
    def __init__(self, embedding_size=256, number_iterations=20):
        self.number_iterations = number_iterations
        self.embedding_size = embedding_size
        self.model = TADW(dim=embedding_size, lamb=0.2)

    def fit(self, X, M):
        self.X = X
        self.M = M
        self.model.fit(self.X, self.M, self.number_iterations)

    def get_embeddings(self):
        return self.model.get_embeddings()

    def get_embeddings_new(self, M):
        return self.model.get_embeddings_new(M)

    def predict(self, i, j):
        u = self.get_embeddings()[i]
        v = self.get_embeddings()[j]
        return 1-scipy.spatial.distance.cosine(u,v)

    def predict_new(self, Mi, Mj):
        [u,v] = self.get_embeddings_new([Mi, Mj])
        return 1-scipy.spatial.distance.cosine(u,v)
