from context import idne

import idne.eval.multi_class_classification
import idne.eval.multi_class_classification_new
import idne.eval.multi_label_classification
import idne.eval.multi_label_classification_new
import idne.eval.link_prediction
import idne.eval.link_prediction_new

import idne.models.idne
import idne.models.lsa
import idne.models.tadw
import idne.models.graph2gauss
import idne.models.gvnrt
import idne.models.cane

import pickle

import logging
import sys
import os
import resource
import pkg_resources
import idne.datasets.io
logger = logging.getLogger()
output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output_link_pred.p")

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
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * 0.9, hard))

def print_scores(model_name, scores, eval_type, dataset_name):
    print("\n\n")
    print(f"{eval_type} => {dataset_name} => {model_name}")
    print(f"{'-':<15} {'             '.join([f'& {int(s*100):02d}%' for s in scores['proportion']])}")
    print(f"{'micro':<15} {'    '.join([f'& {s*100:5.2f}+-{t*100:5.2f}' for s, t in zip(scores['micro'], scores['std'])])}")
    if 'f1' in scores:
        print(f"{'f1':<15} {'    '.join([f'& {s*100:5.2f}+-{t*100:5.2f}' for s, t in zip(scores['f1'], scores['f1_std'])])}")
    #if 'c' in scores:
    #    print(f"{'c':<15} {'        '.join([f'& {c:8.2f}' for c in scores['c']])}")

def main():
    multi_class_dataset_names =  []
    multi_label_dataset_names = []
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "idne", "resources")
    filenames = [f for f in os.listdir(input_dir) if f[-4:] == ".npz"]
    dataset_names = [filename[:-4] for filename in filenames]
    for dn in dataset_names:
        if "stackexchange" in dn or "stackoverflow" in dn or "mathoverflow" in dn:
            multi_label_dataset_names.append(dn)
        else:
            multi_class_dataset_names.append(dn)

    multi_class_dataset_names = ["nyt"] #"cora", "nyt"]
    multi_label_dataset_names = [] #"gaming.stackexchange.com", "travel.stackexchange.com"]

    models = {
        'lsa': idne.models.lsa.Model(),
        'tadw': idne.models.tadw.Model(),
        'idne': idne.models.idne.Model(),
        'graph2gauss': idne.models.graph2gauss.Model(),
        'gvnrt': idne.models.gvnrt.Model(),
        #'cane': idne.models.cane.Model()
    }

    proportions_classification = [0.02, 0.04, 0.06, 0.08, 0.10]
    proportions_multi_classification = [0.1, 0.2, 0.3, 0.4, 0.5]
    proportions_link_prediction = [0.5]
    proportions_classification_new = [0.8]
    proportions_link_prediction_new = [0.8]

    #proportions_classification = [0.02]
    #proportions_multi_classification = [0.1]
    #proportions_classification_new = [0.5]
    #proportions_link_prediction = [0.1]

    n_trials_classif = 1
    n_trials_link_new = 1
    random_state = None

    logger.info(f"multi_class_dataset_names: {multi_class_dataset_names}")
    logger.info(f"multi_label_dataset_names: {multi_label_dataset_names}")
    logger.info(f"proportions_classification: {proportions_classification}")
    logger.info(f"proportions_multi_classification: {proportions_multi_classification}")
    logger.info(f"proportions_classification_new: {proportions_classification_new}")
    logger.info(f"proportions_link_prediction: {proportions_link_prediction}")
    logger.info(f"proportions_link_prediction_new: {proportions_link_prediction_new}")
    logger.info(f"n_trials_classif: {n_trials_classif}")
    logger.info(f"n_trials_link_new: {n_trials_link_new}")
    logger.info(f"random_state: {random_state}")
    logger.info(f"models: {models.keys()}")

    scores_saved = dict()
    for model_name, model in models.items():

        for dataset_name in multi_label_dataset_names:
            adjacency_matrix, texts, labels, labels_mask, tags = idne.datasets.io.load_multi_label_dataset(
                dataset_name)
            scores = idne.eval.link_prediction.evaluate(
                model,
                adjacency_matrix,
                texts,
                proportions_link_prediction,
                random_state=random_state,
                n_trials=n_trials_link_new
            )
            print_scores(model_name, scores, "LINK_PREDICTION", dataset_name)
            scores_saved[f"LINK_PREDICTION__{dataset_name}__{model_name}"] = scores
            pickle.dump(scores_saved, open(output_path, "wb"))

        for dataset_name in multi_class_dataset_names:
            adjacency_matrix, texts, labels, labels_mask = idne.datasets.io.load_multi_class_dataset(dataset_name)
            scores = idne.eval.link_prediction.evaluate(
                model,
                adjacency_matrix,
                texts,
                proportions_link_prediction,
                random_state=random_state,
                n_trials=n_trials_link_new
            )
            print_scores(model_name, scores, "LINK_PREDICTION", dataset_name)
            scores_saved[f"LINK_PREDICTION__{dataset_name}__{model_name}"] = scores
            pickle.dump(scores_saved, open(output_path, "wb"))


        for dataset_name in multi_class_dataset_names:
            adjacency_matrix, texts, labels, labels_mask = idne.datasets.io.load_multi_class_dataset(dataset_name)
            scores = idne.eval.multi_class_classification.evaluate(
                model,
                adjacency_matrix,
                texts,
                labels,
                labels_mask,
                proportions_classification,
                random_state=random_state,
                n_trials=n_trials_classif
            )
            print_scores(model_name, scores, "MULTI_CLASS_CLASSIFICATION", dataset_name)
            scores_saved[f"MULTI_CLASS_CLASSIFICATION__{dataset_name}__{model_name}"] = scores
            pickle.dump(scores_saved, open(output_path, "wb"))

        for dataset_name in multi_label_dataset_names:
            adjacency_matrix, texts, labels, labels_mask, tags = idne.datasets.io.load_multi_label_dataset(dataset_name)
            scores = idne.eval.multi_label_classification.evaluate(
                model,
                adjacency_matrix,
                texts,
                labels,
                labels_mask,
                proportions_multi_classification,
                random_state=random_state,
                n_trials=n_trials_classif
            )
            print_scores(model_name, scores, "MULTI_LABEL_CLASSIFICATION", dataset_name)
            scores_saved[f"MULTI_LABEL_CLASSIFICATION__{dataset_name}__{model_name}"] = scores
            pickle.dump(scores_saved, open(output_path, "wb"))
        
        for dataset_name in multi_label_dataset_names:
            adjacency_matrix, texts, labels, labels_mask, tags = idne.datasets.io.load_multi_label_dataset(dataset_name)
            scores = idne.eval.link_prediction_new.evaluate(
                model,
                adjacency_matrix,
                texts,
                proportions_link_prediction_new,
                random_state=random_state,
                n_trials=n_trials_link_new
            )
            print_scores(model_name, scores, "LINK_PREDICTION_NEW", dataset_name)
            scores_saved[f"LINK_PREDICTION_NEW__{dataset_name}__{model_name}"] = scores
            pickle.dump(scores_saved, open(output_path, "wb"))
        for dataset_name in multi_class_dataset_names:
            adjacency_matrix, texts, labels, labels_mask = idne.datasets.io.load_multi_class_dataset(dataset_name)
            scores = idne.eval.link_prediction_new.evaluate(
                model,
                adjacency_matrix,
                texts,
                proportions_link_prediction_new,
                random_state=random_state,
                n_trials=n_trials_link_new
            )
            print_scores(model_name, scores, "LINK_PREDICTION_NEW", dataset_name)
            scores_saved[f"LINK_PREDICTION_NEW__{dataset_name}__{model_name}"] = scores
            pickle.dump(scores_saved, open(output_path, "wb"))


        for dataset_name in multi_class_dataset_names:
            adjacency_matrix, texts, labels, labels_mask = idne.datasets.io.load_multi_class_dataset(dataset_name)
            scores = idne.eval.multi_class_classification_new.evaluate(
                model,
                adjacency_matrix,
                texts,
                labels,
                labels_mask,
                proportions_classification_new,
                random_state=random_state,
                n_trials=n_trials_link_new
            )
            print_scores(model_name, scores, "MULTI_CLASS_CLASSIFICATION_NEW", dataset_name)
            scores_saved[f"MULTI_CLASS_CLASSIFICATION_NEW__{dataset_name}__{model_name}"] = scores
            pickle.dump(scores_saved, open(output_path, "wb"))


        for dataset_name in multi_label_dataset_names:
            adjacency_matrix, texts, labels, labels_mask, tags = idne.datasets.io.load_multi_label_dataset(dataset_name)
            scores = idne.eval.multi_label_classification_new.evaluate(
                model,
                adjacency_matrix,
                texts,
                labels,
                labels_mask,
                proportions_classification_new,
                random_state=random_state,
                n_trials=n_trials_link_new
            )
            print_scores(model_name, scores, "MULTI_LABEL_CLASSIFICATION_NEW", dataset_name)
            scores_saved[f"MULTI_LABEL_CLASSIFICATION_NEW__{dataset_name}__{model_name}"] = scores
            pickle.dump(scores_saved, open(output_path, "wb"))



if __name__ == '__main__':
    memory_limit() # Limitates maximun memory usage to half
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)


"""
for dataset_name in multi_label_dataset_names:
    adjacency_matrix, texts, labels, labels_mask, tags = idne.datasets.io.load_multi_label_dataset(dataset_name)
    scores = idne.eval.link_prediction.evaluate(
        model,
        adjacency_matrix,
        texts,
        proportions_link_prediction,
        random_state=random_state,
        n_trials=n_trials_link_new
    )
    print_scores(model_name, scores, "LINK_PREDICTION", dataset_name)
    scores_saved[f"LINK_PREDICTION__{dataset_name}__{model_name}"] = scores
    pickle.dump(scores_saved, open(output_path, "wb"))

for dataset_name in multi_class_dataset_names:
    adjacency_matrix, texts, labels, labels_mask = idne.datasets.io.load_multi_class_dataset(dataset_name)
    scores = idne.eval.link_prediction.evaluate(
        model,
        adjacency_matrix,
        texts,
        proportions_link_prediction,
        random_state=random_state,
        n_trials=n_trials_link_new
    )
    print_scores(model_name, scores, "LINK_PREDICTION", dataset_name)
    scores_saved[f"LINK_PREDICTION__{dataset_name}__{model_name}"] = scores
    pickle.dump(scores_saved, open(output_path, "wb"))
"""