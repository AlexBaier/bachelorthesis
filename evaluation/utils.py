import json
import random
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
        'ts+distknn(k=5)': '#82a67d',  # greyish green
        'ts+distknn(k=10)': '#02d8e9',  # aqua blue
        'ts+distknn(k=15)': '#0a481e',  # pine green
        'ts+distknn(k=20)': '#0504aa',  # royal blue
        'ts+linproj': '#ff9408',  # tangerine
        'ts+pwlinproj(c=25)': '#840000',  # dark red
        'ts+pwlinproj(c=50)': '#80013f',  # wine
    }
    default = '#e50000'  # red
    return colors.get(algo, default)


# copied from https://gist.github.com/adewes/5884820
def get_random_color(pastel_factor=0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0, 1.0) for _ in [1, 2, 3]]]


# copied from https://gist.github.com/adewes/5884820
def color_distance(c1, c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1, c2)])


# copied from https://gist.github.com/adewes/5884820
def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color
