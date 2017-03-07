import logging

from gensim.models import Word2Vec
import numpy as np


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    class_id_path = '../data/algorithm_io/class_ids-20161107.txt'
    model_path = '../data/algorithm_io/simple_sentence_model_final-20161107'
    output_path = '../data/algorithm_io/simple_sentence_class_embeddings-20161107'

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
