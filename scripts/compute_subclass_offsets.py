import logging
import time

import gensim
import numpy

from algorithm.sentence_gen import SentenceIterator


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model_path = '../data/algorithm_io/simple_sentence_model-20161107'
    triple_sentences_path = '../data/algorithm_io/simple_sentences-20161107.txt'
    offsets_output_path = '../data/algorithm_io/subclass_offsets-20161107.csv'
    progress_report_interval = 500

    model = gensim.models.Word2Vec.load(model_path)  # type: gensim.models.Word2Vec
    model.delete_temporary_training_data()

    c = 0
    start_time = time.time()

    logging.log(level=logging.INFO, msg='begin offset computation')
    with open(offsets_output_path, mode='w') as f:
        for subclass, superclass in map(lambda r: (r[0], r[2]),
                                        filter(lambda s: s[1] == 'P279',
                                               SentenceIterator(triple_sentences_path))):
            try:
                superclass_vec = model[superclass]
                subclass_vec = model[subclass]
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


if __name__ == '__main__':
    main()
