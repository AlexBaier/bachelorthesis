import json
import logging
from typing import List, Tuple

import numpy as np

from evaluation.data_sample import MultiLabelSample


def load_config(config_path: str)->dict:
    with open(config_path) as f:
        config = json.load(f)
    return config


def load_training_data(training_data_path: str)->List[MultiLabelSample[str]]:
    training_samples = list()  # type: List[MultiLabelSample[str]]
    with open(training_data_path) as f:
        for idx, r in enumerate(f):
            if idx == 0:
                continue
            training_samples.append(MultiLabelSample.from_csv(r, col_sep=','))
    logging.log(level=logging.INFO, msg='loaded training samples')
    return training_samples


def load_test_inputs(test_data_path: str)->List[str]:
    test_inputs = list()  # type: List[str]
    with open(test_data_path) as f:
        for idx, r in enumerate(f):
            if idx == 0:
                continue
            test_inputs.append(MultiLabelSample.from_csv(r, col_sep=',').input_arg)
    logging.log(level=logging.INFO, msg='loaded test inputs')
    return test_inputs


def load_embeddings_and_labels(embeddings_path: str)->Tuple[np.array, List[str]]:
    class_ids = list()
    embeddings = list()
    with open(embeddings_path) as f:
        for cid, embedding in map(lambda l: l.strip().split(';'), f):
            embedding = np.array(embedding.strip('[').strip(']').strip().split(), dtype=np.float32)
            class_ids.append(cid)
            embeddings.append(embedding)
    embeddings = np.array(embeddings)  # type: np.array
    logging.log(level=logging.INFO, msg='loaded embeddings and labels')
    return embeddings, class_ids
