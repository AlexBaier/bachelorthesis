import json
import logging
from typing import Dict, List

import numpy as np

from evaluation.data_sample import MultiLabelSample
from evaluation.statistics import get_accuracy, get_true_positive_count, get_valid_test_input_count
from evaluation.utils import load_test_data


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    predictions_path = paths_config['execution results']
    taxonomic_overlap_path = paths_config['local taxonomic overlaps']
    test_data_path = paths_config['test data']
    evaluation_output = paths_config['evaluation']

    algorithms = ['ts+distknn(k=5)', 'ts+distknn(k=10)', 'ts+distknn(k=15)', 'ts+distknn(k=20)',
                  'ts+linproj', 'ts+pwlinproj(c=25)', 'ts+pwlinproj(c=50)']
    round_to = 5

    golds = load_test_data(test_data_path)  # type: List[MultiLabelSample]
    logging.log(level=logging.INFO, msg='loaded gold standard')

    predictions = dict()  # type: Dict[str, Dict[str, str]]
    for algorithm in algorithms:
        with open(predictions_path.format(algorithm)) as f:
            predictions[algorithm] = dict((u, p) for u, p in map(lambda l: l.strip().split(','), f))
        logging.log(level=logging.INFO, msg='loaded predictions of {}'.format(algorithm))

    overlaps = dict()  # type: Dict[str, List[float]]
    for algorithm in algorithms:
        with open(taxonomic_overlap_path.format(algorithm)) as f:
            overlaps[algorithm] = list(map(float, f.readline().strip().split(',')))
        logging.log(level=logging.INFO, msg='loaded taxonomic overlaps of {}'.format(algorithm))

    test_samples = dict()  # type: Dict[str, int]
    tp_counts = dict()  # type: Dict[str, int]
    accuracies = dict()  # type: Dict[str, float]
    avg_overlaps = dict()  # type: Dict[str, float]

    for algorithm in algorithms:
        test_samples[algorithm] = get_valid_test_input_count(predictions[algorithm], golds)

        tp_counts[algorithm] = get_true_positive_count(predictions[algorithm], golds)
        logging.log(level=logging.INFO, msg='computed TP count and accuracy for {}'.format(algorithm))

        accuracies[algorithm] = np.round(get_accuracy(predictions[algorithm], golds), decimals=round_to)
        logging.log(level=logging.INFO, msg='computed accuracy for {}'.format(algorithm))

        avg_overlaps[algorithm] = np.round(sum(overlaps[algorithm])/len(overlaps[algorithm]), decimals=round_to)
        logging.log(level=logging.INFO, msg='computed average taxonomic overlap for {}'.format(algorithm))

    with open(evaluation_output, mode='w') as f:
        f.write(','.join(['algorithm',
                          'test samples',
                          'TPs',
                          'accuracy',
                          'average taxonomic overlap']) + '\n')
        for algorithm in algorithms:
            row = [
                algorithm,
                str(test_samples[algorithm]),
                str(tp_counts[algorithm]),
                str(accuracies[algorithm]),
                str(avg_overlaps[algorithm])
            ]
            f.write(','.join(row) + '\n')


if __name__ == '__main__':
    main()
