from typing import Callable, List, Tuple

import numpy as np

from evaluation.data_sample import MultiLabelSample


def map_to_knn_training_input(training_samples: List[MultiLabelSample[str]], id2embedding: Callable[[str], np.array])\
        ->Tuple[np.array, np.array]:
    objects = list()
    labels = list()
    for sample in training_samples:
        obj = id2embedding(sample.input_arg)
        for label in sample.possible_outputs:
            objects.append(obj)
            labels.append(label)
    objects = np.array(objects, dtype=np.float32)
    labels = np.array(labels, dtype=np.str)
    return objects, labels


def map_to_proj_training_input(training_samples: List[MultiLabelSample[str]], id2embedding: Callable[[str], np.array])\
        ->List[Tuple[np.array, np.array]]:
    training_data = list()
    for sample in training_samples:
        obj = id2embedding(sample.input_arg)
        for label in sample.possible_outputs:
            training_data.append((obj, id2embedding(label)))
    return training_data
