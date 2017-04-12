import json
import logging
import time

import numpy

from evaluation.utils import load_embeddings_and_labels


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = 'triple sentence sgns'

    with open('algorithm_config.json') as f:
        algorithm_config = json.load(f)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    embeddings_path = algorithm_config[model]['embeddings path']
    subclass_relations_path = paths_config['subclass of relations']
    offsets_output_path = paths_config['subclass offsets']
    progress_report_interval = 500

    embeddings, labels = load_embeddings_and_labels(embeddings_path)

    id2embedding = dict(zip(labels, embeddings))

    superclasses = dict()
    with open(subclass_relations_path) as f:
        for r in map(lambda l: l.strip().split(','), f):
            if len(r) == 1:
                superclasses[r[0]] = set()
            else:
                superclasses[r[0]] = set(r[1:])
    logging.log(level=logging.INFO, msg='loaded superclasses')

    c = 0
    start_time = time.time()

    logging.log(level=logging.INFO, msg='begin offset computation')
    with open(offsets_output_path, mode='w') as f:
        for sub, sups in superclasses.items():
            for sup in sups:
                try:
                    superclass_vec = id2embedding[sup]
                    subclass_vec = id2embedding[sub]
                except KeyError as e:
                    logging.log(logging.DEBUG, msg='Class not found in vocabulary: {}'.format(e))
                    continue
                offset = superclass_vec - subclass_vec
                f.write(';'.join([sub, sup, numpy.array2string(offset).replace('\n', ' ')]) + '\n')
                c += 1
                if c % progress_report_interval == 0:
                    logging.log(level=logging.INFO, msg='offsets computed: {}'.format(c))
    duration = time.time() - start_time

    logging.log(level=logging.INFO, msg='computation finished in {} seconds. computed {} offsets'.format(duration, c))
    logging.log(level=logging.INFO, msg='wrote offsets to {}'.format(offsets_output_path))


if __name__ == '__main__':
    main()
