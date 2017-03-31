import json
import logging
from typing import Dict, List

from data_analysis.dumpio import JSONDumpReader
from data_analysis.utils import get_subclass_of_ids
from evaluation.statistics import get_mean_squared_error, get_near_hits, get_true_positive_count, get_f1_score
from evaluation.utils import load_embeddings_and_labels, load_test_data


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    config_path = '../algorithm_config.json'
    predictions_path = '../evaluation/results_{}-20161107.csv'
    test_data_path = '../evaluation/test_data-20161107.csv'
    classes_path = '../data/classes-20161107'

    evaluation_output = '../evaluation/hybrid_evaluation-20161107.csv'

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

    class_ids = set()
    for gold in golds:
        class_ids.update(gold.possible_outputs)
    for algorithm in algorithms:
        for _, v in predictions[algorithm].items():
            class_ids.add(v)
    logging.log(level=logging.INFO, msg='successors of {} classes required'.format(len(class_ids)))

    succ_nodes = dict()  # type: Dict[str, List[str]]
    count = 0
    for obj in JSONDumpReader(classes_path):
        if obj['id'] not in class_ids:
            continue
        succ_nodes[obj['id']] = list(get_subclass_of_ids(obj))
        count += 1
        if count % 500 == 0:
            logging.log(logging.INFO, msg='successors progress: {}'.format(100.0*float(count)/len(class_ids)))
    logging.log(level=logging.INFO, msg='successors retrieved')

    total_count = len(golds)
    tp_counts = dict()
    mses = dict()
    underspec_counts = dict()
    overspec_counts = dict()
    same_par_counts = dict()
    near_hit_ratios = dict()
    f1_scores = dict()

    for algorithm in algorithms:
        tp_counts[algorithm] = get_true_positive_count(predictions[algorithm], golds)
        logging.log(level=logging.INFO, msg='computed TP count and accuracy for {}'.format(algorithm))
        mses[algorithm] = get_mean_squared_error(predictions[algorithm], golds,
                                                 id2embedding[config['combinations'][algorithm]['sgns']], round_to=5)
        logging.log(level=logging.INFO, msg='computed MSE for {}'.format(algorithm))
        underspec, overspec, same_par = get_near_hits(succ_nodes, predictions[algorithm], golds)
        underspec_counts[algorithm] = underspec
        overspec_counts[algorithm] = overspec
        same_par_counts[algorithm] = same_par
        near_hit_ratios[algorithm] = float(underspec_counts[algorithm]
                                           + overspec_counts[algorithm] + same_par_counts[algorithm]) / total_count
        logging.log(level=logging.INFO, msg='computed NHR for {}'.format(algorithm))
        f1_scores[algorithm] = get_f1_score(predictions[algorithm], golds, round_to=5)
        logging.log(level=logging.INFO, msg='computed precision, recall and F1 for {}'.format(algorithm))
        logging.log(level=logging.INFO, msg='evaluated {}'.format(algorithm))

    with open(evaluation_output, mode='w') as f:
        f.write(','.join(['algorithm',
                          'total',
                          'TPs',
                          'MSE',
                          'underspecialized',
                          'overspecialized',
                          'same_parent',
                          'NHR',
                          'F1']) + '\n')
        for algorithm in algorithms:
            row = [
                algorithm,
                str(total_count),
                str(tp_counts[algorithm]),
                str(mses[algorithm]),
                str(underspec_counts[algorithm]),
                str(overspec_counts[algorithm]),
                str(same_par_counts[algorithm]),
                str(near_hit_ratios[algorithm]),
                str(f1_scores[algorithm])
            ]
            f.write(','.join(row) + '\n')


if __name__ == '__main__':
    main()
