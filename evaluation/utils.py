import json

import numpy as np
from typing import List, Tuple

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
    return training_samples


def load_test_data(test_data_path: str)->List[MultiLabelSample]:
    test_samples = list()  # type: List[MultiLabelSample]
    with open(test_data_path) as f:
        for idx, r in enumerate(f):
            if idx == 0:
                continue
            test_samples.append(MultiLabelSample.from_csv(r, col_sep=','))
    return test_samples


def load_test_inputs(test_data_path: str)->List[str]:
    return [sample.input_arg for sample in load_test_data(test_data_path)]


def load_embeddings_and_labels(embeddings_path: str)->Tuple[np.array, List[str]]:
    class_ids = list()
    embeddings = list()
    with open(embeddings_path) as f:
        for cid, embedding in map(lambda l: l.strip().split(';'), f):
            embedding = np.array(embedding.strip('[').strip(']').strip().split(), dtype=np.float32)
            class_ids.append(cid)
            embeddings.append(embedding)
    embeddings = np.array(embeddings)  # type: np.array
    return embeddings, class_ids


def algo2color(algo):
    colors = {
        'ts+kriknn(k=2&r=1)': '#75bbfd',  # sky blue
        'ts+kriknn(k=5&r=1)': '#8fff9f',  # mint green
        'ts+kriknn(k=10&r=1)': '#137e6d',  # blue green
        'ts+kriknn(k=15&r=1)': '#0504aa',  # royal blue
        'ts+kriknn(k=5&r=10)': '#82a67d',  # greyish green
        'ts+kriknn(k=15&r=10)': '#02d8e9',  # aqua blue
        'ts+distknn(k=15)': '#0a481e',  # pine green
        'ts+linproj': '#ff9408',  # tangerine
        # 'ts+pwlinproj(c=5)': '',
        # 'ts+pwlinproj(c=10)': '',
        # 'ts+pwlinproj(c=20)': '',
        'ts+pwlinproj(c=30)': '#840000',  # dark red
        # 'ts+pwlinproj(c=50)': ''
    }
    default = '#e50000'  # red
    return colors.get(algo, default)
