import logging
logger = logging.getLogger()
import numpy as np
import scipy.sparse
import scipy.spatial.distance
import sklearn.metrics.pairwise
import sklearn.preprocessing

# TODO: general config of w_local etc.

# TF Vectorizers

def get_tf_dictionary(dictionary):
    tf_matrix = scipy.sparse.csr_matrix((dictionary.num_docs, dictionary.num_words))
    for k, seq in enumerate(dictionary.docs_seqs):
        tf_matrix += scipy.sparse.csr_matrix(
            (np.ones(dictionary.docs_lens[k]),
             (k * np.ones(dictionary.docs_lens[k]),
              seq)),
            shape=(dictionary.num_docs, dictionary.num_words)
        )

    #tf_matrix = sklearn.preprocessing.normalize(tf_matrix, norm='l1', axis=1)
    return tf_matrix


def get_tf_1(dictionary, doc):
    seq = dictionary.get_sequence(doc)
    tf_vector = scipy.sparse.csr_matrix(
        (np.ones(len(seq)),
         (np.zeros(len(seq)),
          seq)),
        shape=(1, dictionary.num_words)
    )
    #tf_vector = sklearn.preprocessing.normalize(tf_vector, norm='l1', axis=1)
    return tf_vector


def get_tf_N(dictionary, docs):
    tf_matrix = scipy.sparse.csr_matrix((len(docs), dictionary.num_words))
    for k, doc in enumerate(docs):
        seq = dictionary.get_sequence(doc)
        tf_matrix += scipy.sparse.csr_matrix(
            (np.ones(len(seq)),
             (
                np.ones(len(seq))*k,
                seq)),
            shape=(len(docs), dictionary.num_words)
        )
    #tf_matrix = sklearn.preprocessing.normalize(tf_matrix, norm='l1', axis=1)
    return tf_matrix


# TFIDF Vectorizer

def get_tfidf_dictionary(dictionary):
    tfidf_matrix = scipy.sparse.csr_matrix((dictionary.num_docs, dictionary.num_words))
    for k, seq in enumerate(dictionary.docs_seqs):
        indices, counts = np.unique(seq, return_index=False, return_inverse=False, return_counts=True)
        length = len(indices)
        w_local = counts/counts.sum()
        w_global = np.log(dictionary.num_docs / dictionary.df[indices])
        tfidf_values = w_local * w_global
        tfidf_matrix += scipy.sparse.csr_matrix(
            (tfidf_values,
             (k * np.ones(length),
              indices)),
            shape=(dictionary.num_docs, dictionary.num_words)
        )
    tfidf_matrix = sklearn.preprocessing.normalize(tfidf_matrix, norm='l2', axis=1)
    return tfidf_matrix


def get_tfidf_1(dictionary, doc):
    seq = dictionary.get_sequence(doc)
    indices, counts = np.unique(seq, return_index=False, return_inverse=False, return_counts=True)
    length = len(indices)
    w_local = counts/counts.sum()
    w_global = np.log(dictionary.num_docs / dictionary.df[indices])
    tfidf_values = w_local * w_global
    tfidf_vector = scipy.sparse.csr_matrix(
        (tfidf_values,
         (np.zeros(length),
          indices)),
        shape=(1, dictionary.num_words)
    )
    tfidf_vector = sklearn.preprocessing.normalize(tfidf_vector, norm='l2', axis=1)
    return tfidf_vector


def get_tfidf_N(dictionary, docs):
    tfidf_matrix = scipy.sparse.csr_matrix((len(docs), dictionary.num_words))
    for k, doc in enumerate(docs):
        seq = dictionary.get_sequence(doc)
        indices, counts = np.unique(seq, return_index=False, return_inverse=False, return_counts=True)
        length = len(indices)
        w_local = counts/counts.sum()
        w_global = np.log(dictionary.num_docs / dictionary.df[indices])
        tfidf_values = w_local * w_global
        tfidf_matrix += scipy.sparse.csr_matrix(
            (tfidf_values,
             (k * np.ones(length),
              indices)),
            shape=(len(docs), dictionary.num_words)
        )
    tfidf_matrix = sklearn.preprocessing.normalize(tfidf_matrix, norm='l2', axis=1)
    return tfidf_matrix



























