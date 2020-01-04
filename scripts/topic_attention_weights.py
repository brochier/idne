from context import idne

import idne.models.idne
import idne.eval.visualization
import numpy as np
import logging
import sys
import resource
import pkg_resources
import idne.datasets.io
from sklearn.preprocessing import normalize
import idne.preprocessing.text.dictionary
import os



np.set_printoptions(precision=2)
logger = logging.getLogger()
def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * 0.75, hard))

def get_html(word_labels, best_tops, best_vals,  t1, t2):
    html = ""
    for i, w in enumerate(word_labels):
        if best_tops[i] == t1:
            html += f'<span style="background-color:rgb({255*(1-best_vals[i])}, 255, 255)">{w}</span> '
        elif best_tops[i] == t2:
            html += f'<span style="background-color:rgb(255, 255, {255*(1-best_vals[i])})">{w}</span> '
        else:
            html += f'<span style="background-color:rgb(255, 255, 255)">{w}</span> '
    return html

from idne.preprocessing.text.tokenizer import *

def tokenize(documents):
    tokenized_documents = list()
    for d in documents:
        s = stringify(d).lower()
        s = strip_tags(s)
        s = strip_non_alphanum(s)
        s = deaccent(s)
        s = strip_punctuation(s)
        #s = remove_stopwords(s)
        s = strip_multiple_whitespaces(s)
        t = s.split()
        t = [w for w in t if len(w) > 1]
        tokenized_documents.append(t)
    return tokenized_documents

def main():
    dataset_name = "travel.stackexchange.com"
    #adjacency_matrix, texts, labels, labels_mask = idne.datasets.io.load_multi_class_dataset(dataset_name)
    #number_incuding_points = len(np.unique(labels))
    adjacency_matrix, texts, labels, labels_mask, _ = idne.datasets.io.load_multi_label_dataset(dataset_name)
    number_incuding_points = labels.shape[1]

    vocab = idne.preprocessing.text.dictionary.Dictionary(texts, tokenizer=tokenize)

    model = idne.models.idne.Model(embedding_size=256,
                                      number_iterations=5e3,
                                      number_incuding_points = 8,
                                      number_negative=1
                                      )

    model.fit(adjacency_matrix, texts, vocab=vocab)

    current_folder = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_folder, os.path.pardir, "output", "travel_attention_weights.html")

    doc = "<html><head></head><body>"


    print("Plot")
    model.plot_direct_topics()
    aw_list = list()
    word_labels_list = list()
    top_list = list()
    max_top = list()
    second_max_top = list()
    third_max_top = list()
    for text in texts:
        aw, word_labels = model.raw_attention(text)
        aw_list.append(aw)
        word_labels_list.append(word_labels)
        top = aw.sum(axis=1) / aw.sum()
        top_list.append(top)
        max_top.append(np.max(top))
        second_max_top.append(top[np.argsort(top)[::-1][1]])
        third_max_top.append(top[np.argsort(top)[::-1][2]])


    top_specific = np.argsort(max_top)[::-1]
    top_specific_filtered = list()
    for i, ts in enumerate(top_specific):
        k = top_specific[i]
        #if third_max_top[top_specific[i]] < 0.10:
        if (max_top[k] - second_max_top[k]) < (second_max_top[k] - third_max_top[k]):
            top_specific_filtered.append(k)

    for i in np.array(top_specific_filtered, dtype=np.int)[0:50]:
        print(top_list[i], word_labels_list[i])
        best_2_topics = np.argsort(top_list[i])[::-1][:2]
        best_tops = np.argmax(aw_list[i], axis=0)
        best_vals = np.max(aw_list[i], axis=0)
        text_html = get_html(word_labels_list[i], best_tops, best_vals,  best_2_topics[0],  best_2_topics[1])
        doc += f"<hr> <p>Topic distribution: {top_list[i]} => yellow best topic, blue second best:</p>"
        doc += f"<p>{text_html}</p>"

    doc += "</body></html>"

    with open(output_file, 'w') as f:
        f.write(doc)
    return True





if __name__ == '__main__':
    memory_limit() # Limitates maximun memory usage to half
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)