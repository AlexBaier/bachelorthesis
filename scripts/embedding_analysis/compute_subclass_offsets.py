import json
import logging
import time

import numpy

from algorithm.sequence_gen import Sequences
from evaluation.utils import load_embeddings_and_labels


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = 'triple sentence sgns'
    sequence_gen = 'triple sentences'

    with open('algorithm_config.json') as f:
        algorithm_config = json.load(f)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    embeddings_path = algorithm_config[model]
    triple_sentences_path = paths_config[sequence_gen]
    offsets_output_path = paths_config['subclass offsets']
    progress_report_interval = 500

    embeddings, labels = load_embeddings_and_labels(embeddings_path)

    id2embedding = dict(zip(labels, embeddings))

    c = 0
    start_time = time.time()

    logging.log(level=logging.INFO, msg='begin offset computation')
    with open(offsets_output_path, mode='w') as f:
        for subclass, superclass in map(lambda r: (r[0], r[2]),
                                        filter(lambda s: s[1] == 'P279',
                                               Sequences([triple_sentences_path]))):
            try:
                superclass_vec = id2embedding[superclass]
                subclass_vec = id2embedding[subclass]
            except KeyError as e:
                logging.log(logging.WARNING, msg='Class not found in vocabulary: {}'.format(e))
                continue
            offset = superclass_vec - subclass_vec
            f.write(';'.join([subclass, superclass, numpy.array2string(offset).replace('\n', ' ')]) + '\n')
            c += 1
            if c % progress_report_interval == 0:
                logging.log(level=logging.INFO, msg='offsets computed: {}'.format(c))
    duration = time.time() - start_time

    logging.log(level=logging.INFO, msg='computation finished in {} seconds. computed {} offsets'.format(duration, c))
    logging.log(level=logging.INFO, msg='wrote offsets to {}'.format(offsets_output_path))


if __name__ == '__main__':
    main()
