import json
import logging

from evaluation.execution import NO_INPUT_EMBEDDING
from evaluation.statistics import get_local_taxonomic_overlap
from evaluation.utils import load_test_data


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    algorithm = 'ts+distknn(k=15)'

    with open('paths_config.json') as f:
        config = json.load(f)

    predictions_path = config['execution results']
    gold_path = config['test data']
    subclass_of_path = config['subclass of relations']
    output_path = config['local taxonomic overlaps']

    golds = load_test_data(gold_path)
    logging.log(level=logging.INFO, msg='loaded golds')

    with open(predictions_path.format(algorithm)) as f:
        predictions = dict((u, p) for u, p in filter(lambda s: s[1] != NO_INPUT_EMBEDDING,
                                                     map(lambda l: l.strip().split(','), f)))
    logging.log(level=logging.INFO, msg='loaded predictions of {}'.format(algorithm))

    superclasses = dict()
    with open(subclass_of_path) as f:
        for r in map(lambda l: l.strip().split(','), f):
            if len(r) == 1:
                superclasses[r[0]] = set()
            else:
                superclasses[r[0]] = set(r[1:])
    logging.log(level=logging.INFO, msg='loaded superclasses')

    local_precisions = list()
    for idx, gold in enumerate(filter(lambda g: predictions.get(g.input_arg, None), golds)):
        local_precision = \
            get_local_taxonomic_overlap(predictions[gold.input_arg], gold.possible_outputs, superclasses)
        local_precisions.append(local_precision)
    logging.log(level=logging.INFO, msg='computed local precisions for {}'.format(algorithm))

    logging.log(level=logging.INFO, msg='average overlap: {}'.format(sum(local_precisions)/len(local_precisions)))

    with open(output_path.format(algorithm), mode='w') as f:
        f.write(','.join(map(str, local_precisions)) + '\n')
    logging.log(level=logging.INFO, msg='wrote local overlaps to {}'.format(output_path.format(algorithm)))


if __name__ == '__main__':
    main()
