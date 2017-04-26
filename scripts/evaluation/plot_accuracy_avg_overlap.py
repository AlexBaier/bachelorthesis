import json
import logging

import matplotlib.pyplot as plt
import numpy as np


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    k_neighbors = np.array([5, 10, 15, 20])
    c_clusters = np.array([1, 25, 50])

    with open('paths_config.json') as f:
        config = json.load(f)

    evaluation_path = config['evaluation']
    knn_plot_output = config['knn evaluation plot']
    linproj_plot_output = config['linproj evaluation plot']

    accuracies = dict()
    overlaps = dict()
    with open(evaluation_path) as f:
        f.readline()
        for r in map(lambda l: l.strip().split(','), f):
            accuracies[r[0]] = float(r[3])
            overlaps[r[0]] = float(r[4])
    logging.log(level=logging.INFO, msg='loaded evaluation results')

    # plot ts+distknn(k=?) accuracies and overlaps
    plt.figure(1)
    plt.clf()

    plt.ylim([0, 1])
    knn_accuracies = np.array([accuracies['ts+distknn(k={})'.format(k)] for k in k_neighbors])
    knn_overlaps = np.array([overlaps['ts+distknn(k={})'.format(k)] for k in k_neighbors])
    acc_rect = plt.bar(k_neighbors-0.2, knn_accuracies, width=0.4, color='#15b01a')
    over_rect = plt.bar(k_neighbors+0.2, knn_overlaps, width=0.4, color='#0343df')
    plt.xticks(k_neighbors)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend((acc_rect, over_rect), ('accuracy', 'taxonomic overlap'))
    plt.grid(True)
    plt.savefig(knn_plot_output)
    logging.log(level=logging.INFO, msg='stored knn evaluation plot to {}'.format(knn_plot_output))


if __name__ == '__main__':
    main()
