import logging
logger = logging.getLogger()
import time
import numpy as np
import os
import math
from idne.preprocessing.text.tokenizer import *



def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def tokenize(documents):
    tokenized_documents = list()
    for d in documents:
        s = stringify(d).lower()
        s = strip_tags(s)
        s = strip_non_alphanum(s)
        s = deaccent(s)
        s = strip_punctuation(s)
        s = strip_multiple_whitespaces(s)
        t = s.split()
        t = [w for w in t if len(w) > 0]
        tokenized_documents.append(t)
    return tokenized_documents


def to_id_sequence(doc, words_to_ids, pad=False):
    if pad:
        return [words_to_ids[w] if w in words_to_ids else -1 for w in tokenize([doc])[0]]
    else:
        return [words_to_ids[w] for w in tokenize([doc])[0] if w in words_to_ids]


def to_token_sequence(doc, words_to_ids, pad=False):
    if pad:
        return [w if w in words_to_ids else "_" for w in tokenize([doc])[0]]
    else:
        return [w for w in tokenize([doc])[0] if w in words_to_ids]


class Vocabulary():
    def __init__(self):
        self.ids_to_words = [] # list of words
        self.words_to_ids = {} # dict of words to ids
        self.num_words = 0

    def add_words(self, documents):
        tokenized_docs = tokenize(documents)
        for t in tokenized_docs:
            for w in t:
                if w not in self.words_to_ids:
                    self.words_to_ids[w] = len(self.ids_to_words)
                    self.ids_to_words.append(w)
        self.num_words = len(self.ids_to_words)

    def get_sequences(self, docs, pad=False):
        return np.array([to_id_sequence(doc, self.words_to_ids, pad=pad) for doc in docs], dtype=list)

    def get_sequence(self, doc, pad=False):
        return to_id_sequence(doc, self.words_to_ids, pad=pad)

    def get_tokens(self, doc, pad=False):
        return to_token_sequence(doc, self.words_to_ids, pad=pad)

    def load(self, folder, name):
        logger.info("Loading dictionary...")
        self.ids_to_words, self.words_to_ids, self.num_words = np.load(
            os.path.join(folder, name)
        )
        logger.info("ids_to_words: {0}".format(len(self.ids_to_words)))
        logger.info("words_to_ids: {0}".format(len(self.words_to_ids)))
        logger.info("num_words: {0}".format(self.num_words))
        return self

    def save(self, folder, name):
        data_to_save = [
            self.ids_to_words,
            self.words_to_ids,
            self.num_words
        ]
        np.save(os.path.join(folder, name), data_to_save)






