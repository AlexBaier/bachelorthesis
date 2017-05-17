import logging
from typing import Callable, List, Tuple

import numpy as np

from evaluation.data_sample import MultiLabelSample


def map_to_knn_training_input(training_samples: List[MultiLabelSample[str]], id2embedding: Callable[[str], np.array])\
        ->Tuple[np.array, np.array]:
    objects = list()
    labels = list()
    valid_sample_count = 0
    for sample in training_samples:
        try:
            obj = id2embedding(sample.input_arg)
            for label in sample.possible_outputs:
                objects.append(obj)
                labels.append(label)
            valid_sample_count += 1
        except KeyError as e:
            logging.log(level=logging.DEBUG, msg='no embedding for {}'.format(e))
    logging.log(level=logging.INFO, msg='generated kNN training samples: {}/{} samples used'
                .format(len(training_samples), valid_sample_count))
    objects = np.array(objects, dtype=np.float32)
    labels = np.array(labels, dtype=np.str)
    return objects, labels


def map_to_proj_training_input(training_samples: List[MultiLabelSample[str]], id2embedding: Callable[[str], np.array])\
        ->Tuple[List[Tuple[np.array, np.array]], List[str]]:
    training_data = list()
    labels = list()
    valid_sample_count = 0
    for sample in training_samples:
        try:
            obj = id2embedding(sample.input_arg)
        except KeyError as e:
            logging.log(level=logging.DEBUG, msg='no embedding for {}'.format(e))
            continue
        for label in sample.possible_outputs:
            try:
                superclass = id2embedding(label)
                training_data.append((obj, superclass))
                labels.append(label)
            except KeyError as e:
                logging.log(level=logging.DEBUG, msg='no embedding for {}'.format(e))
        valid_sample_count += 1
    logging.log(level=logging.INFO, msg='generated lin proj training samples: {}/{} samples used'
                .format(len(training_samples), valid_sample_count))
    return training_data, labels


def map_to_baseline_training_input(training_samples: List[MultiLabelSample[str]], id2embedding: Callable[[str], np.array])\
        ->List[str]:
    return [output for sample in training_samples for output in sample.possible_outputs]
