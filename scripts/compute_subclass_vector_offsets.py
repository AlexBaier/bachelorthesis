import logging
import sqlite3

import gensim
import numpy


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model_path = '../data/algorithm_io/simple_sentence_model-20161107'
    edges_db_path = '../data/algorithm_io/edges-20161107.sqlite3'
    output_path = '../data/algorithm_io/subclass_offsets-20161107'

    conn = sqlite3.connect(edges_db_path)
    cursor = conn.cursor()
    model = gensim.models.Word2Vec.load(model_path)

    relations = cursor.execute('SELECT s, t FROM edges WHERE r = 279')
    logging.log(level=logging.INFO, msg='executed sql query')

    with open(output_path, mode='w') as f:
        c = 0
        for subclass, superclass in relations.fetchall():
            subclass = 'Q' + str(subclass)
            superclass = 'Q' + str(superclass)
            superclass_vec = model.get(superclass, None)
            subclass_vec = model.get(subclass, None)
            if not superclass_vec or not subclass_vec:
                continue
            offset = superclass_vec - subclass_vec
            x_axis = numpy.zeros(offset.shape)
            x_axis[0] = 1
            norm = numpy.linalg.norm(offset)
            normalized_offset = offset / norm
            angle = numpy.arccos(numpy.clip(numpy.dot(normalized_offset, x_axis), -1.0, 1.0))
            f.write(','.join([subclass, superclass, str(norm.tostring()), str(angle.tostring())]) + '\n')
            c += 1
            if c % 500 == 0:
                logging.log(level=logging.INFO, msg='offsets calculated: {}'.format(c))


if __name__ == '__main__':
    main()
