import logging
from typing import Callable, Dict, List, Tuple, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from evaluation.data_sample import MultiLabelSample


def get_true_positive_count(predictions: Dict[str, str], golds: List[MultiLabelSample])->int:
    tp_count = 0
    for gold in golds:
        if predictions[gold.input_arg] in gold.possible_outputs:
            tp_count += 1
    return tp_count


def get_mean_squared_error(predictions: Dict[str, str], golds: List[MultiLabelSample],
                           id2embedding: Callable[[str], np.array])->float:
    n = float(len(golds))
    mse = (1/n * np.sum((1.0 - get_prediction_gold_cosine_similarities(predictions, golds, id2embedding))**2))
    return mse


def get_prediction_gold_cosine_similarities(predictions: Dict[str, str], golds: List[MultiLabelSample],
                                            id2embedding: Callable[[str], np.array])->np.array:
    similarities = np.zeros(len(golds), dtype=np.float64)
    for idx, gold in enumerate(golds):
        prediction = id2embedding(predictions[gold.input_arg]).reshape(1, -1)
        golds = list()
        for output in gold.possible_outputs:
            try:
                golds.append(id2embedding(output))
            except KeyError:
                continue
        if len(golds) == 0:
            continue
        golds = np.array(golds)
        if golds.shape[0] == 1:
            golds = golds.reshape(1, -1)
        similarities[idx] = np.max(cosine_similarity(prediction, golds))
    return similarities


def get_near_hits(succ_nodes: Dict[str, List[str]], predictions: Dict[str, str], golds: List[MultiLabelSample])\
        ->Tuple[int, int, int]:
    """
    Computes taxonomic neighborhood relations between prediction and gold standard.
    The following cases are counted:
    underspecialized: prediction is superclass of gold standard.
    overspecialized: prediction is subclass of gold standard.
    :param succ_nodes: 
    :param predictions: 
    :param golds: 
    :return: (underspecialized, overspecialized, same_parent)
    """
    underspecialized = 0  # type: int
    overspecialized = 0  # type: int
    same_parent = 0  # type: int

    prediction_gold_pairs = [(predictions[gold.input_arg], gold.possible_outputs) for gold in golds]

    n = len(prediction_gold_pairs)

    def is_underspecialized(prediction, classes):
        for c in classes:
            c_outs = succ_nodes.get(c, list())
            if prediction in c_outs:
                return True
        return False

    def have_same_parent(prediction_outs, classes):
        for c in classes:
            c_outs = succ_nodes.get(c, list())
            if set(prediction_outs).intersection(set(c_outs)):
                return True
        return False

    for idx, (pred, gold) in enumerate(prediction_gold_pairs):
        if idx + 1 % 500 == 0:
            logging.log(level=logging.INFO, msg='near hit progress: {}%'
                        .format(100.0*float(idx+1)/n))

        if pred in gold:
            continue

        p_outs = succ_nodes.get(pred, list())

        if set(p_outs).intersection(set(gold)):
            overspecialized += 1
            continue

        if is_underspecialized(pred, gold):
            underspecialized += 1
            continue

        if have_same_parent(p_outs, gold):
            same_parent += 1
            continue

    return underspecialized, overspecialized, same_parent


def get_weighted_precision_recall_f1(predictions: Dict[str, str], golds: List[MultiLabelSample])\
        ->Tuple[float, float, float]:
    """
    Probably based on
    http://text-analytics101.rxnlp.com/2014/10/computing-precision-and-recall-for.html (accessed: 2017-03-29)
    :param predictions: 
    :param golds: 
    :return: (precision, recall, F1)
    """
    labels = list()  # type: List[str]

    # get all relevant labels:
    for gold in golds:
        labels.append(predictions[gold.input_arg])
        labels.extend(gold.possible_outputs)
    # remove duplicates
    labels = list(set(labels))
    n = len(labels)
    # map from label to idx
    label2idx = dict((labels[idx], idx) for idx in range(n))

    # initialize confusion matrix
    confusion = np.zeros((n, n), dtype=np.int32)
    # fill confusion matrix
    for gold in golds:
        prediction = predictions[gold.input_arg]
        if prediction in gold.input_arg:
            confusion[label2idx[prediction]][label2idx[prediction]] += 1
        else:
            confusion[label2idx[prediction]][label2idx[gold.possible_outputs[0]]] += 1

    # compute precision and recall for labels, and total weighted precision and recall
    precisions = dict()
    recalls = dict()

    precision = 0.0
    recall = 0.0
    for label in labels:
        tp = confusion[label2idx[label]][label2idx[label]]  # type: int
        total_gold = np.sum(confusion[:, label2idx[label]])  # type: int
        total_predicted = np.sum(confusion[label2idx[label], :])  # type: int

        # compute precision
        if total_predicted > 0:
            precisions[label] = float(tp) / float(total_predicted)
        else:
            precisions[label] = 0.0

        # compute recall
        if total_gold > 0:
            recalls[label] = float(tp) / float(total_gold)
        else:
            recalls[label] = 0.0

        precision += total_predicted * precisions[label]
        recall += total_predicted * recalls[label]

    precision /= len(golds)
    recall /= len(golds)

    # compute final weighted F1-score
    f1_score = 2.0 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score
