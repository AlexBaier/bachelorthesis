"""
For some reason the execution of algorithms using graph walk sentence embeddings fails because of memory
issues. Specifically the NearestNeighbors.predict method fails to terminate, since all processes
wait and never complete because no memory for this to occur is ever free.

This script will compare the memory usage of simple sentence embeddings and graph walk sentence embeddings.
The results are printed to stdout.

Note:
    It turns out there is no difference in memory usage, which should be expected, since both sets of embeddings
    should have the same shape: (number_of_classes, embedding_size).
Output:
    triple sentence sgns:
        bytes:1559401200
        shape:(1299501, 300)
    graph walk sentence sgns:
        bytes:1559401200
        shape:(1299501, 300)
    difference: 0
"""
import json

import logging

from evaluation.utils import load_embeddings_and_labels


def main():
    config_path = '../algorithm_config.json'

    embedding_types = [
        'triple sentence sgns',
        'graph walk sentence sgns'
    ]

    with open(config_path) as f:
        config = json.load(f)

    bytes_used = dict()
    shapes = dict()

    for embedding_type in embedding_types:
        embeddings, _ = load_embeddings_and_labels(config[embedding_type]['embeddings path'])
        bytes_used[embedding_type] = embeddings.nbytes
        shapes[embedding_type] = embeddings.shape
        logging.log(level=logging.INFO, msg='loaded {}'.format(embedding_types))

    for embedding_type in embedding_types:
        print('{}:'.format(embedding_type))
        print('\tbytes:{}'.format(bytes_used[embedding_type]))
        print('\tshape:{}'.format(shapes[embedding_type]))
    print('difference: {}'.format(bytes_used['graph walk sentence sgns']-bytes_used['triple sentence sgns']))

if __name__ == '__main__':
    main()
