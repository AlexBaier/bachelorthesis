import json
import logging
from typing import Dict, Set
from typing import List, Tuple

import numpy as np

import algorithm.classification as alg
from algorithm.utils import map_to_knn_training_input, map_to_proj_training_input
from evaluation.data_sample import MultiLabelSample


def execute_combined_algorithms(combined_algorithms: Set[str], config_path: str, training_data_path: str,
                                test_data_path: str)->Dict[str, List[Tuple[str, str]]]:

    config = load_config(config_path)  # type: dict

    training_samples = load_training_data(training_data_path)  # type: List[MultiLabelSample[str]]

    test_inputs = load_test_inputs(test_data_path)  # type: List[str]

    results = dict()  # type: Dict[str, List[Tuple[str, str]]

    for combined_algorithm in combined_algorithms:

        try:
            components = config['combinations'][combined_algorithm]
        except KeyError:
            raise UnknownAlgorithmError(combined_algorithm)

        try:
            sgns = components['sgns']
        except KeyError:
            raise MissingComponentError('sgns', combined_algorithm)
        try:
            classification = components['classification']
        except KeyError:
            raise MissingComponentError('classification', combined_algorithm)

        try:
            sgns_config = config[sgns]
        except KeyError:
            raise UnknownComponentError('sgns', sgns)
        try:
            classification_config = config[classification]
        except KeyError:
            raise UnknownComponentError('classification', classification)

        try:
            embeddings_path = sgns_config['embeddings path']
        except KeyError:
            raise MissingParameterError('embeddings path', sgns)

        embeddings, class_ids = load_embeddings_and_labels(embeddings_path)

        result = execute_classification(algorithm=classification,
                                        config=classification_config,
                                        embeddings=embeddings,
                                        class_ids=class_ids,
                                        training_samples=training_samples,
                                        test_inputs=test_inputs
                                        )
        results[combined_algorithm] = result
        logging.log(level=logging.INFO, msg='completed execution of combined algorithm "{}"'.format(combined_algorithm))
    return results


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


def execute_classification(algorithm: str, config: dict,
                           embeddings: np.array, class_ids: List[str],
                           training_samples: List[MultiLabelSample[str]],
                           test_inputs: List[str])->List[Tuple[str, str]]:
    id2idx = dict()  # type: Dict[str, int]
    for idx in range(len(class_ids)):
        id2idx[class_ids[idx]] = idx

    def id2embedding(entity_id: str) -> np.array:
        return embeddings[id2idx[entity_id]]

    classifier = None  # type: alg.Classifier

    if algorithm in ['kri-knn', 'distance-knn']:
        mapping = map_to_knn_training_input
    elif algorithm in ['linear projection', 'piecewise linear projection']:
        mapping = map_to_proj_training_input
    else:
        raise NotImplementedClassifierError(algorithm)

    training_input = mapping(training_samples, id2embedding)
    logging.log(level=logging.INFO, msg='prepared training data')

    if algorithm == 'kri-knn':
        classifier = alg.KRINKNNClassifier()
    elif algorithm == 'distance-knn':
        try:
            neighbors = config['neighbors']
        except KeyError:
            raise MissingParameterError('neighbors', algorithm)
        classifier = alg.DistanceKNNClassifier(neighbors=neighbors)
    elif algorithm == 'linear projection':
        classifier = alg.LinearProjectionClassifier(embedding_size=embeddings.shape[1],
                                                    embeddings=embeddings,
                                                    labels=class_ids)
    elif algorithm == 'piecewise linear projection':
        try:
            clusters = config['clusters']
        except KeyError:
            raise MissingParameterError("clusters", algorithm)
        classifier = alg.PiecewiseLinearProjectionClassifier(embedding_size=embeddings.shape[1],
                                                             clusters=clusters,
                                                             embeddings=embeddings,
                                                             labels=class_ids)
    logging.log(level=logging.INFO, msg='initialized {} classifier'.format(algorithm))

    classifier.train(training_input)
    logging.log(level=logging.INFO, msg='trained {} classifier'.format(algorithm))

    test_input_matrix = np.array(list(map(lambda t: id2embedding(t), test_inputs[:10])))
    labels = classifier.classify(test_input_matrix)
    results = list()  # type: List[Tuple[str, str]]
    for i in range(len(test_inputs)):
        results.append((test_inputs[i], labels[i]))
    logging.log(level=logging.INFO, msg='executed classification for all test inputs')

    return results


class UnknownAlgorithmError(LookupError):
    def __init__(self, algorithm):
        self.strerror = 'combined algorithm with name {} not found in config'.format(algorithm)
        self.args = {algorithm}


class MissingComponentError(LookupError):
    def __init__(self, component_type, algorithm):
        self.strerror = '{} component is not defined for {}'.format(component_type, algorithm)
        self.args = {component_type, algorithm}


class UnknownComponentError(LookupError):
    def __init__(self, component_type, component):
        self.strerror = 'classification component "{}" not found in config'.format(component_type, component)
        self.args = {component_type, component}


class MissingParameterError(LookupError):
    def __init__(self, parameter, component):
        self.strerror = 'parameter "{}" missing in {}'.format(parameter, component)
        self.args = {parameter, component}


class NotImplementedClassifierError(Exception):
    def __init__(self, classification_component):
        self.strerror = 'classification method "{}" does not exist'.format(classification_component)
        self.args = {classification_component}
