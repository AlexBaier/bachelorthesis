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


def get_prediction_gold_taxonomic_relations(self, edge_db_conn: sqlite3.Connection, predictions: Dict[str, str],
                            golds: List[MultiLabelSample], max_dist_to_common_parent: int=3)\
        ->Tuple[List[int], List[int], List[int], List[str]]:
    """
    Get taxonomic relations of misclassifications
    :return: distance between prediction and gold in case of (underspecialized, overspecialized, wrong branch)
    """
    # TODO: implement
    underspecialized = list()
    overspecialized = list()
    wrong_branch = list()
    distance_exceeded = list()

    return underspecialized, overspecialized, wrong_branch, distance_exceeded

