import logging
logger = logging.getLogger()
import time
from idne.preprocessing.text.tokenizer import *
import numpy as np
import os
import math


def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])


# TODO: change min_df to min_wf
class Dictionary():
    def __init__(self, documents=None, tokenizer=tokenize, min_df = 1, max_df_ratio = 1, forced_vocab=None, save_memory=False):
        # Important attributes
        self.ids_to_words = list() # list of words
        self.df = list() # number of document-occurrence of each word
        self.wf = list() # number of single-occurrence of each word
        self.words_to_ids = dict() # dict word -> id
        self.num_words = 0 # number of words in corpus
        self.num_docs = 0 # number of documents
        # Optional attributes
        self.docs_seqs = list() # list of sequences of id
        self.docs_lens = list() # length of each document
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.tokenizer = tokenizer

        if documents is not None:
            if forced_vocab is not None:
                self.forced_vocab = set()
                f_vocab = self.tokenizer(forced_vocab)
                for fd in f_vocab:
                    for fw in fd:
                        self.forced_vocab.add(fw)
            else:
                self.forced_vocab = set()

            logger.debug("Starting tokenizing documents...")
            if save_memory:
                N = len(documents)
                K = 0
                S = min(100, N)
                for i in range(N):
                    if i >= K-1:
                        logger.debug(f'{( (i+1) / N * 100):.0f}% tokenized')
                        K += N / S
                    documents[i] = self.tokenizer([documents[i]])[0]
            else:
                documents = self.tokenizer(documents)

            logger.debug("Starting building dictionary...")
            self.__build(documents)

            logger.debug("Starting pruning dictionary...")
            self.__prune(documents)

            self.df = np.array(self.df, dtype=np.int)
            self.wf = np.array(self.wf, dtype=np.int)
            self.ids_to_words = np.array(self.ids_to_words, dtype=str)
            for i, doc_seq in enumerate(self.docs_seqs):
                self.docs_seqs[i] = np.array(doc_seq, dtype=np.int)
            self.docs_seqs = np.array(self.docs_seqs, dtype=list)
            self.docs_lens = np.array(self.docs_lens, dtype=np.int)

            #logger.debug("Memory usage after prune:")
            #logger.debug(f"ids_to_words: {convert_size(self.ids_to_words.nbytes)}")
            #logger.debug(f"docs_seqs: {convert_size(self.docs_seqs.nbytes)}")

    def load(self, folder, name):
        logger.debug("Loading dictionary...")
        self.ids_to_words, self.df, self.words_to_ids, self.num_words, self.num_docs = np.load(
            os.path.join(folder, name)
        )
        logger.debug("ids_to_words: {0}".format(len(self.ids_to_words)))
        logger.debug("df: {0}".format(len(self.df)))
        logger.debug("words_to_ids: {0}".format(len(self.words_to_ids)))
        logger.debug("num_words: {0}".format(self.num_words))
        logger.debug("num_docs: {0}".format(self.num_docs))
        return self


    def save(self, folder, name):
        data_to_save = [
            self.ids_to_words,
            self.df,
            self.words_to_ids,
            self.num_words,
            self.num_docs
        ]
        np.save(os.path.join(folder, name), data_to_save)

    def __build(self, documents):
        start_time = time.time()
        N = len(documents)
        K = 0
        S = min(100, N)
        for i, document in enumerate(documents):
            if i >= K - 1:
                logger.debug(f'{((i + 1) / N * 100):.0f}% built')
                K += N / S
            self.num_docs += 1
            #self.docs_lens.append(len(document))
            sequence = list()
            w_set = set()
            for word in document:
                if word not in self.words_to_ids:
                    self.words_to_ids[word] = self.num_words
                    self.num_words += 1
                    self.ids_to_words.append(word)
                    self.wf.append(0)
                    self.df.append(0)
                sequence.append(self.words_to_ids[word])
                self.wf[self.words_to_ids[word]] += 1
                if word not in w_set:
                    self.df[self.words_to_ids[word]] += 1
                    w_set.add(word)
            #self.docs_seqs.append(sequence)

        logger.debug("Dictionary build in {0} seconds".format(time.time() - start_time))
        logger.debug("num_documents={0}, num_words={1}".format(
            self.num_docs,
            self.num_words
        ))
        logger.debug("df => min_occ={0}, max_occ={1}, mean_occ={2}, std_occ={3} ".format(
            np.min(self.df),
            np.max(self.df),
            np.mean(self.df),
            np.std(self.df)
        ))
        logger.debug("wf => min_occ={0}, max_occ={1}, mean_occ={2}, std_occ={3} ".format(
            np.min(self.wf),
            np.max(self.wf),
            np.mean(self.wf),
            np.std(self.wf)
        ))


    def __prune(self, documents):
        start_time = time.time()
        min_mask = np.array(self.wf, dtype=int) >= self.min_df
        max_mask = (np.array(self.df, dtype=float) / self.num_docs) <= self.max_df_ratio
        mask = np.logical_and(min_mask, max_mask)

        forced_words = [self.words_to_ids[w] for w in self.forced_vocab]
        forced_mask = np.zeros(len(self.ids_to_words)).astype(np.bool)
        forced_mask[forced_words] = True

        mask = np.logical_or(mask, forced_mask)

        self.ids_to_words = [w for i,w in enumerate(self.ids_to_words) if mask[i]]
        self.df = [f for i,f in enumerate(self.df) if mask[i]]
        self.wf = [f for i,f in enumerate(self.wf) if mask[i]]
        self.words_to_ids = {w: i for i, w in enumerate(self.ids_to_words)}
        self.num_words = len(self.ids_to_words)

        N = len(documents)
        K = 0
        S = min(100, N)
        for k, doc in enumerate(documents):
            if k >= K - 1:
                logger.debug(f'{((k + 1) / N * 100):.0f}% pruned')
                K += N / S
            new_seq = list()
            for w in doc:
                if w in self.words_to_ids:
                    new_seq.append(self.words_to_ids[w])

            self.docs_lens.append(len(new_seq))
            self.docs_seqs.append(new_seq)

        logger.debug("Dictionary pruned in {0} seconds".format(time.time() - start_time))
        logger.debug("num_documents={0}, num_words={1}".format(
            self.num_docs,
            self.num_words
        ))
        logger.debug("df => min_occ={0}, max_occ={1}, mean_occ={2}, std_occ={3} ".format(
            np.min(self.df),
            np.max(self.df),
            np.mean(self.df),
            np.std(self.df)
        ))
        logger.debug("wf => min_occ={0}, max_occ={1}, mean_occ={2}, std_occ={3} ".format(
            np.min(self.wf),
            np.max(self.wf),
            np.mean(self.wf),
            np.std(self.wf)
        ))
        logger.debug("docs_lens => min_len={0}, max_len={1}, mean_len={2}, std_len={3} ".format(
            np.min(self.docs_lens),
            np.max(self.docs_lens),
            np.mean(self.docs_lens),
            np.std(self.docs_lens)
        ))

    def get_sequence(self, doc):
        doc = self.tokenizer([doc])[0]
        return np.array([self.words_to_ids[w] for w in doc if w in self.words_to_ids], dtype=np.int32)
