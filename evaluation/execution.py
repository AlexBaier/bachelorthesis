import itertools
import logging
import os
import re
from typing import Dict, List, Tuple

import numpy as np

import algorithm.classification as alg
from algorithm.utils import map_to_knn_training_input, map_to_proj_training_input, map_to_baseline_training_input, \
    map_to_neural_network_training_input
from evaluation.data_sample import MultiLabelSample
from evaluation.utils import load_config, load_embeddings_and_labels, load_test_inputs, load_training_data

__MOST_COMMON_REGEX = re.compile(r'^most-common$')
__DIST_KNN_REGEX = re.compile(r'^distance-knn \(k=[1-9][0-9]*\)$')
__PW_LIN_PROJ_REGEX = re.compile(r'^linear projection \(c=[1-9][0-9]*\)$')
__DNN_REGR_REGEX = re.compile(r'^deep neural network \(h=[1-9][0-9]*, n=[1-9][0-9]*\)$')
__CONCAT_NN_REGR_REGEX = re.compile(r'^concat neural network \(act=\w+, net=[1-9][0-9]*, h=[1-9][0-9]*, n=[1-9][0-9]*\)$')

NO_INPUT_EMBEDDING = 'NO_INPUT_EMBEDDING'


def execute_combined_algorithms(combined_algorithms: List[str], config_path: str, training_data_path: str,
                                test_data_path: str, workers)->Dict[str, List[Tuple[str, str]]]:

    config = load_config(config_path)  # type: dict

    training_samples = load_training_data(training_data_path)  # type: List[MultiLabelSample[str]]

    test_inputs = load_test_inputs(test_data_path)  # type: List[str]

    results = dict()  # type: Dict[str, List[Tuple[str, str]]

    for hybrid_algorithm in combined_algorithms:

        try:
            components = config['combinations'][hybrid_algorithm]
        except KeyError as e:
            raise UnknownAlgorithmError(str(e))

        try:
            sgns = components['sgns']
            classification = components['classification']
        except KeyError as e:
            raise MissingComponentError(str(e), hybrid_algorithm)

        try:
            sgns_config = config[sgns]
        except KeyError as e:
            raise UnknownComponentError('sgns', str(e))

        try:
            classification_config = config[classification]
        except KeyError as e:
            raise UnknownComponentError('classification', str(e))

        try:
            embeddings_path = sgns_config['embeddings path']
        except KeyError as e:
            raise MissingParameterError('embeddings path', str(e))

        if embeddings_path:
            embeddings, class_ids = load_embeddings_and_labels(embeddings_path)
        else:
            embeddings = np.zeros((len(test_inputs), 1))
            class_ids = list(itertools.islice(itertools.cycle(test_inputs), len(test_inputs)))

        result = execute_classification(algorithm=classification,
                                        config=classification_config,
                                        embeddings=embeddings,
                                        class_ids=class_ids,
                                        training_samples=training_samples,
                                        test_inputs=test_inputs,
                                        workers=workers
                                        )
        results[hybrid_algorithm] = result
        logging.log(level=logging.INFO, msg='completed execution of combined algorithm "{}"'.format(hybrid_algorithm))
    return results


def execute_classification(algorithm: str, config: dict,
                           embeddings: np.array, class_ids: List[str],
                           training_samples: List[MultiLabelSample[str]],
                           test_inputs: List[str], workers: int)->List[Tuple[str, str]]:
    id2idx = dict()  # type: Dict[str, int]
    for idx in range(len(class_ids)):
        id2idx[class_ids[idx]] = idx

    def id2embedding(entity_id: str) -> np.array:
        return embeddings[id2idx[entity_id]]

    classifier = None  # type: alg.Classifier

    if __DIST_KNN_REGEX.fullmatch(algorithm):
        mapping = map_to_knn_training_input
    elif __PW_LIN_PROJ_REGEX.fullmatch(algorithm):
        mapping = map_to_proj_training_input
    elif __MOST_COMMON_REGEX.fullmatch(algorithm):
        mapping = map_to_baseline_training_input
    elif __DNN_REGR_REGEX.fullmatch(algorithm) or __CONCAT_NN_REGR_REGEX.fullmatch(algorithm):
        mapping = map_to_neural_network_training_input
    else:
        raise NotImplementedClassifierError(algorithm)

    training_input = mapping(training_samples, id2embedding)
    logging.log(level=logging.INFO, msg='prepared training data')

    if __DIST_KNN_REGEX.fullmatch(algorithm):
        try:
            neighbors = config['neighbors']
        except KeyError as e:
            raise MissingParameterError(str(e), algorithm)
        classifier = alg.DistanceKNNClassifier(neighbors=neighbors, n_jobs=workers)
    elif __PW_LIN_PROJ_REGEX.fullmatch(algorithm):
        try:
            clusters = config['clusters']
            sgd_iter = config['sgd iterations']
        except KeyError as e:
            raise MissingParameterError(str(e), algorithm)

        classifier = alg.PiecewiseLinearProjectionClassifier(embedding_size=embeddings.shape[1],
                                                             clusters=clusters,
                                                             sgd_iter=sgd_iter,
                                                             n_jobs=workers)
    elif __MOST_COMMON_REGEX.fullmatch(algorithm):
        classifier = alg.MostCommonClassClassifier()
    elif __DNN_REGR_REGEX.fullmatch(algorithm):
        try:
            n_hidden_layers = config['hidden layers']
            n_hidden_neurons = config['hidden neurons']
            epochs = config['epochs']
            batch_size = config['batch size']
        except KeyError as e:
            raise MissingParameterError(str(e), algorithm)
        try:
            model_path = config['model path'] if os.path.isfile(config['model path']) else None
        except KeyError:
            model_path = None
        classifier = alg.DeepFFRegressionClassifier(embedding_size=embeddings.shape[1],
                                                    n_hidden_layers=n_hidden_layers,
                                                    n_hidden_neurons=n_hidden_neurons,
                                                    epochs=epochs,
                                                    batch_size=batch_size,
                                                    n_jobs=workers,
                                                    model_path=model_path)
    elif __CONCAT_NN_REGR_REGEX.fullmatch(algorithm):
        try:
            activation = config['activation']
            n_networks = config['networks']
            n_hidden_layers = config['hidden layers']
            n_hidden_neurons = config['hidden neurons']
            epochs = config['epochs']
            batch_size = config['batch size']
        except KeyError as e:
            raise MissingParameterError(str(e), algorithm)
        try:
            model_path = config['model path'] if os.path.isfile(config['model path']) else None
        except KeyError:
            model_path = None
        classifier = alg.ConcatFFRegressionClassifier(activation=activation,
                                                      embedding_size=embeddings.shape[1],
                                                      n_networks=n_networks,
                                                      n_hidden_layers=n_hidden_layers,
                                                      n_hidden_neurons=n_hidden_neurons,
                                                      epochs=epochs,
                                                      batch_size=batch_size,
                                                      n_jobs=workers,
                                                      model_path=model_path)
    logging.log(level=logging.INFO, msg='initialized {} classifier'.format(algorithm))

    classifier.train(training_input)
    logging.log(level=logging.INFO, msg='trained {} classifier'.format(algorithm))
    if (__DNN_REGR_REGEX.fullmatch(algorithm) or __CONCAT_NN_REGR_REGEX.fullmatch(algorithm)) \
            and config.get('model path', None):
        classifier.save_to_file(config['model path'])
        logging.info('stored trained model {} to {}'.format(algorithm, config['model path']))

    test_input_matrix = list()
    is_valid_test = list()
    for idx, test_input in enumerate(test_inputs):
        try:
            test_input_matrix.append(id2embedding(test_input))
            is_valid_test.append(True)
        except KeyError as e:
            logging.log(level=logging.DEBUG, msg='missing embedding for test input {}'.format(e))
            test_input_matrix.append(np.zeros(embeddings.shape[1]))
            is_valid_test.append(False)
    logging.log(level=logging.INFO, msg='{}/{} valid test cases'
                .format(np.array(is_valid_test).sum(), len(test_inputs)))
    test_input_matrix = np.array(test_input_matrix)
    labels = classifier.classify(test_input_matrix)
    del classifier

    results = list()  # type: List[Tuple[str, str]]
    for i in range(len(test_inputs)):
        results.append((test_inputs[i], labels[i] if is_valid_test[i] else NO_INPUT_EMBEDDING))
    logging.log(level=logging.INFO, msg='executed classification for all test inputs')

    return results


class UnknownAlgorithmError(LookupError):
    def __init__(self, algorithm):
        self.strerror = 'hybrid algorithm with name {} not found in config'.format(algorithm)
        self.args = {self.strerror}


class MissingComponentError(LookupError):
    def __init__(self, component_type, algorithm):
        self.strerror = '"{}" component is not defined for "{}"'.format(component_type, algorithm)
        self.args = {self.strerror}


class UnknownComponentError(LookupError):
    def __init__(self, component_type, component):
        self.strerror = '{} component "{}" not found in config'.format(component_type, component)
        self.args = {self.strerror}


class MissingParameterError(LookupError):
    def __init__(self, parameter, component):
        self.strerror = 'parameter "{}" missing in "{}"'.format(parameter, component)
        self.args = {self.strerror}


class NotImplementedClassifierError(Exception):
    def __init__(self, classification_component):
        self.strerror = 'classification method "{}" does not exist'.format(classification_component)
        self.args = {self.strerror}
