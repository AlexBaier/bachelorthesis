import json
import logging

from evaluation.execution import NO_INPUT_EMBEDDING
from evaluation.statistics import get_local_taxonomic_overlap
from evaluation.utils import load_test_data


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    algorithms = ['baseline', 'ts+dnn(h=4,n=3600)', 'ts+dnn(h=8,n=2400)', 'ts+concatnn(act=relu,net=20,n=2400)',
                  'ts+concatnn(act=relu,net=10,h=3,n=1200)', 'ts+concatnn(act=relu,net=20,h=3,n=1200)',
                  'ts+concatnn(act=relu,net=30,h=3,n=600)', 'ts+concatnn(act=relu,net=20,h=4,n=1200)',
                  'ts+concatnn(act=relu,net=1,h=3,n=1200)',
                  'ts+distknn(k=5)', 'ts+distknn(k=10)', 'ts+distknn(k=15)', 'ts+distknn(k=20)',
                  'ts+linproj(c=1)', 'ts+linproj(c=25)', 'ts+linproj(c=50)']

    with open('paths_config.json') as f:
        config = json.load(f)

    predictions_path = config['execution results']
    gold_path = config['test data']
    subclass_of_path = config['subclass of relations']
    output_path = config['local taxonomic overlaps']

    golds = load_test_data(gold_path)
    logging.log(level=logging.INFO, msg='loaded golds')

    predictions = dict()
    for algorithm in algorithms:
        with open(predictions_path.format(algorithm)) as f:
            predictions[algorithm] = dict((u, p) for u, p in filter(lambda s: s[1] != NO_INPUT_EMBEDDING,
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

    for algorithm in algorithms:
        local_precisions = list()
        for idx, gold in enumerate(filter(lambda g: predictions[algorithm].get(g.input_arg, None), golds)):
            local_precision = \
                get_local_taxonomic_overlap(predictions[algorithm][gold.input_arg], gold.possible_outputs, superclasses)
            local_precisions.append(local_precision)
        logging.log(level=logging.INFO, msg='computed local precisions for {}'.format(algorithm))

        logging.log(level=logging.INFO,
                    msg='average overlap ({}): {}'.format(algorithm, sum(local_precisions)/len(local_precisions)))

        with open(output_path.format(algorithm), mode='w') as f:
            f.write(','.join(map(str, local_precisions)) + '\n')
        logging.log(level=logging.INFO, msg='wrote local overlaps to {}'.format(output_path.format(algorithm)))


if __name__ == '__main__':
    main()
