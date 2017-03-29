import sqlite3
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from evaluation.data_sample import MultiLabelSample


def get_true_positive_count(predictions: Dict[str, str], golds: List[MultiLabelSample])->int:
    tp_count = 0
    for gold in golds:
        if predictions[gold.input_arg] in gold.possible_outputs:
            tp_count += 1
    return tp_count


def get_true_positive_ratio(predictions: Dict[str, str], golds: List[MultiLabelSample])->float:
    tps = get_true_positive_count(predictions, golds)
    n = len(golds)
    return float(tps)/n


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


def get_near_hits(edge_db_path: str, predictions: Dict[str, str], golds: List[MultiLabelSample])\
        ->Tuple[int, int]:
    """
    Computes taxonomic neighborhood relations between prediction and gold standard.
    The following cases are counted:
    underspecialized: prediction is superclass of gold standard.
    overspecialized: prediction is subclass of gold standard.
    :param edge_db_path: 
    :param predictions: 
    :param golds: 
    :return: (underspecialized, overspecialized)
    """
    underspecialized = 0  # type: int
    overspecialized = 0  # type: int

    pred_gold_pairs = [(predictions[gold.input_arg], gold.possible_outputs)for gold in golds]

    with sqlite3.connect(edge_db_path) as conn:
        cursor = conn.cursor()
        succ_nodes = dict()  # type: Dict[str, List[str]]

        def get_outs(node):
            if not succ_nodes.get(node, None):
                cursor.execute('SELECT * FROM edges WHERE s=?', [int(node[1:])])
                succ_nodes[node] = list(map(lambda e: 'Q{}'.format(e[2]), cursor.fetchall()))
            return succ_nodes[node]

        def is_underspecialized(prediction, classes):
            for c in classes:
                c_outs = get_outs(c)
                if prediction in c_outs:
                    return True
            return False

        for pred, gold in pred_gold_pairs:
            p_outs = get_outs(pred)

            if set(p_outs).intersection(set(gold)):
                overspecialized += 1
                continue

            if is_underspecialized(pred, gold):
                underspecialized += 1
                continue

    return underspecialized, overspecialized
