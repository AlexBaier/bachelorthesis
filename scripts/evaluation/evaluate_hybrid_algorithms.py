import json
import logging

from typing import Dict, List

from evaluation.data_sample import MultiLabelSample
from evaluation.statistics import get_accuracy, get_true_positive_count, get_valid_test_input_count
from evaluation.utils import load_test_data


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    with open('algorithm_config.json') as f:
        algorithm_config = json.load(f)

    predictions_path = paths_config['execution results']
    test_data_path = paths_config['test data']
    evaluation_output = paths_config['evaluation']

    algorithms = [
        'ts+distknn(k=15)',
        'ts+linproj'
    ]

    golds = load_test_data(test_data_path)  # type: List[MultiLabelSample]
    logging.log(level=logging.INFO, msg='loaded gold standard')

    predictions = dict()  # type: Dict[str, Dict[str, str]]
    for algorithm in algorithms:
        with open(predictions_path.format(algorithm)) as f:
            predictions[algorithm] = dict((u, p) for u, p in map(lambda l: l.strip().split(','), f))
        logging.log(level=logging.INFO, msg='loaded predictions of {}'.format(algorithm))

    training_samples = dict()  # type: Dict[str, int]
    test_samples = dict()  # type: Dict[str, int]
    tp_counts = dict()  # type: Dict[str, int]
    accuracies = dict()  # type: Dict[str, int]

    for algorithm in algorithms:
        training_samples[algorithm] = algorithm_config['combinations'][algorithm]['training samples']
        test_samples[algorithm] = get_valid_test_input_count(predictions[algorithm], golds)

        tp_counts[algorithm] = get_true_positive_count(predictions[algorithm], golds)
        logging.log(level=logging.INFO, msg='computed TP count and accuracy for {}'.format(algorithm))

        accuracies[algorithm] = get_accuracy(predictions[algorithm], golds)
        logging.log(level=logging.INFO, msg='computed accuracy for {}'.format(algorithm))

    with open(evaluation_output, mode='w') as f:
        f.write(','.join(['algorithm',
                          'training_samples',
                          'test samples',
                          'TPs',
                          'accuracy']) + '\n')
        for algorithm in algorithms:
            row = [
                algorithm,
                str(training_samples[algorithm]),
                str(test_samples[algorithm]),
                str(tp_counts[algorithm]),
                str(accuracies[algorithm])
            ]
            f.write(','.join(row) + '\n')


if __name__ == '__main__':
    main()
