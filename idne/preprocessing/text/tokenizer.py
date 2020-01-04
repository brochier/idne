import idne.preprocessing.text.stop_words as stop_words
import idne.preprocessing.text.regex as regex
import unidecode

import logging
logger = logging.getLogger()

def stringify(text):
    if isinstance(text, str):
        return text
    return str(text, 'utf8', errors='strict')


def remove_stopwords(s):
    return " ".join(w for w in s.split() if w not in stop_words.STOPWORDS)


def strip_punctuation(s):
    return regex.RE_PUNCT.sub(" ", s)


def strip_tags(s):
    return regex.RE_TAGS.sub(" ", s)


def strip_short(s, minsize=3):
    return " ".join(e for e in s.split() if len(e) >= minsize)


def strip_numeric(s):
    return regex.RE_NUMERIC.sub("", s)


def strip_non_alphanum(s):
    return regex.RE_NONALPHA.sub(" ", s)


def deaccent(s):
    return unidecode.unidecode(s)


def strip_multiple_whitespaces(s):
    return regex.RE_WHITESPACE.sub(" ", s)


def split_alphanum(s):
    s = regex.RE_AL_NUM.sub(r"\1 \2", s)
    return regex.RE_NUM_AL.sub(r"\1 \2", s)


def tokenize(documents):
    tokenized_documents = list()
    for d in documents:
        s = stringify(d).lower()
        s = strip_tags(s)
        s = strip_non_alphanum(s)
        s = deaccent(s)
        s = strip_punctuation(s)
        s = remove_stopwords(s)
        s = strip_multiple_whitespaces(s)
        t = s.split()
        t = [w for w in t if len(w) > 1]
        tokenized_documents.append(t)
    return tokenized_documents