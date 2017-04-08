import json
import logging

import numpy as np
from gensim.models import Word2Vec


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    model = 'triple sentence sgns'

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    with open('algorithm_config.json') as f:
        algorithm_config = json.load(f)

    class_id_path = paths_config['class ids']
    model_path = paths_config[model]
    output_path = algorithm_config[model]

    model = Word2Vec.load(model_path)

    with open(class_id_path) as f:
        class_ids = [l.strip() for l in f]
    logging.log(level=logging.DEBUG, msg='class count: {}'.format(len(class_ids)))

    with open(output_path, mode='w') as f:
        c = 0
        for cid in class_ids:
            try:
                f.write(';'.join([cid, np.array2string(model[cid]).replace('\n', ' ')]) + '\n')
                c += 1
            except KeyError as e:
                logging.log(level=logging.DEBUG, msg='no embedding for {}'.format(e))
        logging.log(level=logging.INFO, msg='wrote embeddings for {} out of {} classes'.format(c, len(class_ids)))


if __name__ == '__main__':
    main()
