import json
import logging

from evaluation.statistics import get_mean_squared_error, get_near_hits, get_true_positive_count
from evaluation.utils import load_embeddings_and_labels, load_test_data


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    config_path = '../algorithm_config.json'
    predictions_path = '../evaluation/results_{}-20161107.csv'
    test_data_path = '../evaluation/test_data-20161107.csv'
    edge_db_path = '../data/algorithm_io/edges-20161107.sqlite3'

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

    total_count = len(golds)
    tp_counts = dict()
    tp_ratios = dict()
    mses = dict()
    underspec_counts = dict()
    overspec_counts = dict()
    near_hit_ratios = dict()
    for algorithm in algorithms:
        tp_counts[algorithm] = get_true_positive_count(predictions[algorithm], golds)
        tp_ratios[algorithm] = float(tp_counts[algorithm]) / total_count
        logging.log(level=logging.INFO, msg='computed TPR for {}'.format(algorithm))
        mses[algorithm] = get_mean_squared_error(predictions[algorithm], golds,
                                                 id2embedding[config['combinations'][algorithm]['sgns']])
        logging.log(level=logging.INFO, msg='computed MSE for {}'.format(algorithm))
        underspec, overspec = get_near_hits(edge_db_path, predictions[algorithm], golds)
        underspec_counts[algorithm] = underspec
        overspec_counts[algorithm] = overspec
        near_hit_ratios[algorithm] = float(tp_counts[algorithm] + underspec_counts[algorithm]
                                           + overspec_counts[algorithms]) / total_count
        logging.log(level=logging.INFO, msg='computed NHR for {}'.format(algorithm))
        logging.log(level=logging.INFO, msg='evaluated {}'.format(algorithm))

    with open(evaluation_output, mode='w') as f:
        f.write(','.join(['algorithm', 'TPs', 'TPR', 'MSE', 'underspecialized', 'overspecialized', 'NHR', 'F1']) + '\n')
        for algorithm in algorithms:
            row = [
                algorithm,
                str(tp_counts[algorithm]),
                str(tp_ratios[algorithm]),
                str(mses[algorithm]),
                str(underspec_counts[algorithm]),
                str(overspec_counts[algorithm]),
                str(near_hit_ratios[algorithm]),
                'not implemented'
            ]
            f.write(','.join(row) + '\n')


if __name__ == '__main__':
    main()
