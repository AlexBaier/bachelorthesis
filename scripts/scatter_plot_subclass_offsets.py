import logging

import matplotlib.pyplot as plt
import numpy


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    offsets_path = '../data/algorithm_io/subclass_offsets-20161107.csv'
    plot_output = '../data/plots/2d_subclass_offsets.png'

    xs = list()
    ys = list()
    c = 0

    with open(offsets_path) as f:
        for subclass, superclass, coordinates in map(lambda s: s.strip().split(';'), f):
            # i don't know why i would do this
            coordinates = numpy.array(coordinates.strip('[').strip(']').strip().split(), dtype=numpy.float32)
            xs.append(coordinates[0])
            ys.append(coordinates[1])
            c += 1
            if c % 500 == 0:
                logging.log(level=logging.INFO, msg='points already added: {}'.format(c))
    logging.log(level=logging.INFO, msg='create plot with {} points'.format(c))
    plt.scatter(xs, ys, s=0.1)
    logging.log(level=logging.INFO, msg='save plot to {}'.format(plot_output))
    plt.savefig(plot_output)

if __name__ == '__main__':
    main()
