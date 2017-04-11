import numpy as np
from typing import Dict, List, Set, Tuple

from evaluation.data_sample import MultiLabelSample
from evaluation.execution import NO_INPUT_EMBEDDING


def get_true_positive_count(predictions: Dict[str, str], golds: List[MultiLabelSample])->int:
    tp_count = 0
    for gold in golds:
        if predictions[gold.input_arg] in gold.possible_outputs:
            tp_count += 1
    return tp_count


def get_valid_test_input_count(predictions: Dict[str, str], golds: List[MultiLabelSample])->int:
    missing_input_count = len(list(filter(lambda v: v == NO_INPUT_EMBEDDING, predictions.values())))
    n = len(golds)

    return n - missing_input_count


def get_accuracy(predictions: Dict[str, str], golds: List[MultiLabelSample], round_to: int=3)->float:
    tps = get_true_positive_count(predictions, golds)
    n = get_valid_test_input_count(predictions, golds)

    accuracy = float(tps) / float(n)

    return np.round(accuracy, decimals=round_to)


def get_local_taxonomic_overlap(computed: str, references: List[str], superclasses: Dict[str, Set[str]])->float:
    """
    See Dellschaft,Staab: On how to perform a gold standard based evaluation of ontology learning. 2006.
    => Taxonomic precision and recall.
    Taxonomic upwards cotopy is used as characteristic extract.
    :param computed: 
    :param references: 
    :param superclasses: 
    :return: 
    """
    computed_cotopy = get_semantic_one_directional_cotopy(computed, superclasses)
    reference_cotopies = list()
    for reference in references:
        reference_cotopy = get_semantic_one_directional_cotopy(reference, superclasses)
        reference_cotopies.append(reference_cotopy)

    taxonomic_overlap = max(len(computed_cotopy.intersection(reference_cotopy)) /
                            len(computed_cotopy.union(reference_cotopy)) for reference_cotopy in reference_cotopies)

    return taxonomic_overlap


def get_semantic_one_directional_cotopy(node: str, successors: Dict[str, Set[str]])->Set[str]:
    """
    :param node: 
    :param successors: 
    :return: semantic cotopy in one direction given by successors, typically set of all transitive sub- or superclasses
        of node.
    """
    trans_superclasses = set()
    q = {node}

    while len(q) > 0:
        current = q.pop()
        trans_superclasses.add(current)
        q.update(successors.get(current, set()).difference(trans_superclasses))

    return trans_superclasses


def get_confusion_matrix(predictions: Dict[str, str], golds: List[MultiLabelSample])\
        ->Tuple[np.array, Dict[str, int]]:
    """ 
    :param predictions: 
    :param golds: 
    :return: confusion matrix, mapping from label to matrix index
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
        if prediction in gold.possible_outputs:
            confusion[label2idx[prediction]][label2idx[prediction]] += 1
        else:
            confusion[label2idx[prediction]][label2idx[gold.possible_outputs[0]]] += 1

    return confusion, label2idx
