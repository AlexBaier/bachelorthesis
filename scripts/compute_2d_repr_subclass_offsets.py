import logging
import sqlite3

import gensim
import numpy
from sklearn.decomposition import PCA

from algorithm.sentence_gen import SentenceIterator


def main():
    # 2017-03-01 21:43:03,116 : INFO : computed 1630923 offsets in 2258.3208146095276 seconds
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model_path = '../data/algorithm_io/simple_sentence_model-20161107'
    triple_sentences_path = '../data/algorithm_io/simple_sentences-20161107.txt'
    output_path = '../data/algorithm_io/subclass_offsets-20161107.csv'
    max_offset_amount = 4e5
    progress_report_interval = 500

    model = gensim.models.Word2Vec.load(model_path)

    offsets = list()
    class_pairs = list()

    c = 0
    logging.log(level=logging.INFO, msg='begin offset computation')
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
        offsets.append(offset)
        class_pairs.append((subclass, superclass))
        c += 1
        if c % progress_report_interval == 0:
            logging.log(level=logging.INFO, msg='offset computation progress: {}'.format(float(c)/max_offset_amount))
        if c >= max_offset_amount:
            break

    del model

    logging.log(level=logging.INFO, msg='execute pca')
    offsets = numpy.array(offsets)
    pca = PCA(n_components=2)
    pca.fit(offsets)
    offsets = pca.transform(offsets)
    logging.log(level=logging.INFO, msg='completed pca and transformed offsets')
    logging.log(level=logging.INFO, msg='offsets dimension: {}'.format(offsets.shape))

    with open(output_path, mode='w') as f:
        for i in range(len(class_pairs)):
            f.write(';'.join([class_pairs[i][0], class_pairs[i][1], numpy.array2string(offsets[i])]) + '\n')


if __name__ == '__main__':
    main()
