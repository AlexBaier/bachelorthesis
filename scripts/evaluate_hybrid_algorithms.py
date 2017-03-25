import json
import logging

from evaluation.statistics import get_mean_squared_error, get_true_positive_ratio
from evaluation.utils import load_embeddings_and_labels, load_test_data


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    config_path = '../algorithm_config.json'
    predictions_path = '../evaluation/results_{}-20161107.csv'
    test_data_path = '../evaluation/test_data-20161107.csv'

    evaluation_output = '../evaluation/hybrid_evaluation.csv'

    algorithms = [
        'baseline',
        'distknn',
        'linproj',
        'pwlinproj'
    ]

    with open(config_path) as f:
        config = json.load(f)
    logging.log(level=logging.INFO, msg='loaded algorithm config')

    golds = load_test_data(test_data_path)
    logging.log(level=logging.INFO, msg='loaded gold standard')

    predictions = dict()
    for algorithm in algorithms:
        with open(predictions_path.format(algorithm)) as f:
            predictions[algorithm] = dict((u, p) for u, p in map(lambda l: l.strip().split(','), f))
        logging.log(level=logging.INFO, msg='loaded predictions of {}'.format(algorithm))

    id2embedding = dict()
    for algorithm in algorithms:
        model = config['combinations'][algorithm]['sgns']
        if model in id2embedding.keys():
            continue
        embeddings, labels = load_embeddings_and_labels(config[model]['embeddings path'])
        id2idx = dict((label, idx) for idx, label in enumerate(labels))
        id2embedding[model] = lambda item_id: embeddings[id2idx[item_id]]
        logging.log(level=logging.INFO, msg='loaded embeddings of {}'.format(model))

    with open(evaluation_output, mode='w') as f:
        f.write(','.join(['algorithm', 'tp_ratio', 'mse']) + '\n')
        for algorithm in algorithms:
            tp_ratio = get_true_positive_ratio(predictions[algorithm], golds)
            mse = get_mean_squared_error(predictions[algorithm], golds,
                                         id2embedding[config['combinations'][algorithm]['sgns']])
            f.write(','.join([algorithm, str(tp_ratio), str(mse)]) + '\n')
            logging.log(level=logging.INFO, msg='evaluated {}'.format(algorithm))


if __name__ == '__main__':
    main()
