import idne.preprocessing.text.dictionary
import idne.preprocessing.text.vectorizers
from sklearn.decomposition import TruncatedSVD
import logging
import numpy as np
import scipy.spatial.distance

logger = logging.getLogger()


class Model:
    def __init__(self, embedding_size = 256):
        self.X = None
        self.M = None
        self.embedding_size = embedding_size

    def fit(self, X, M):
        logger.debug("Building vocab")
        self.vocab = idne.preprocessing.text.dictionary.Dictionary(M, min_df=5, max_df_ratio=0.25)
        logger.debug("Building tfidf vectors")
        tfidf_vectors = idne.preprocessing.text.vectorizers.get_tfidf_dictionary(self.vocab)
        logger.debug("Building svd vectors")
        self.svd = TruncatedSVD(n_components=self.embedding_size)
        self.vectors = self.svd.fit_transform(tfidf_vectors)
        self.embedding_size = self.svd.components_.shape[0]
        logger.debug(f"SVD dimension is: {self.embedding_size}")
        self.X = X

    def get_embeddings(self):
        return self.vectors

    def get_embeddings_new(self, M):
        tfidf = idne.preprocessing.text.vectorizers.get_tfidf_N(self.vocab, M)
        vectors = self.svd.transform(tfidf)
        return vectors

    def predict(self, i, j):
        u = self.get_embeddings()[i]
        v = self.get_embeddings()[j]
        return 1-scipy.spatial.distance.cosine(u, v)

    def predict_new(self, Mi, Mj):
        [u, v] = self.get_embeddings_new([Mi, Mj])
        return 1-scipy.spatial.distance.cosine(u, v)



