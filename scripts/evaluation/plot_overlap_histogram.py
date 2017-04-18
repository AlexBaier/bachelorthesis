import json
import logging

import matplotlib.pyplot as plt

from evaluation.utils import algo2color


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    algorithms = ['ts+distknn(k=5)', 'ts+distknn(k=10)', 'ts+distknn(k=15)', 'ts+distknn(k=20)', 'ts+linproj',
                  'ts+pwlinproj(c=25)', 'ts+pwlinproj(c=50)', 'gw+distknn(k=5)', 'gw+distknn(k=10)', 'gw+distknn(k=15)',
                  'gw+distknn(k=20)', 'gw+linproj']
    n_bins = 30

    with open('paths_config.json') as f:
        config = json.load(f)

    overlaps_path = config['local taxonomic overlaps']
    histogram_path = config['local taxonomic overlap histogram']

    for algorithm in algorithms:
        with open(overlaps_path.format(algorithm)) as f:
            overlaps = list(map(float, f.readline().strip().split(',')))

        plt.figure(1)
        plt.clf()

        plt.hist(x=overlaps, bins=n_bins, label=algorithm, color=algo2color(algorithm), cumulative=True, normed=True)
        plt.title('{}: cumulative local taxonomic overlap histogram'.format(algorithm))
        plt.xlabel('$P[t \leq T]$: local taxonomic overlap')
        plt.ylabel('percentage of predictions')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])

        plt.savefig(histogram_path.format(algorithm))
        logging.log(level=logging.INFO, msg='wrote histogram to {}'.format(histogram_path.format(algorithm)))


if __name__ == '__main__':
    main()
