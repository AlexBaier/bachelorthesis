import json
import logging

import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_overlap_plot(title, xlabel, xticks, accuracies, overlaps, output_path):
    width = np.max(xticks) / 10.0
    offset = width / 2.0

    plt.figure(1)
    plt.clf()
    plt.ylim([0, 1])
    acc_rect = plt.bar(xticks-offset*2, accuracies, width=width, color='#15b01a')
    over_rect = plt.bar(xticks, overlaps, width=width, color='#0343df')
    plt.xticks(xticks)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend((acc_rect, over_rect), ('accuracy', 'taxonomic overlap'))
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel('accuracy/taxonomic overlap')
    plt.title(title)
    plt.savefig(output_path)
    logging.log(level=logging.INFO, msg='stored ts+distknn evaluation plot to {}'.format(output_path))


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    k_neighbors = np.array([5, 10, 15, 20])
    c_clusters = np.array([1, 25, 50])

    with open('paths_config.json') as f:
        config = json.load(f)

    evaluation_path = config['evaluation']
    ts_knn_plot_output = config['ts+distknn evaluation plot']
    ts_linproj_plot_output = config['ts+linproj evaluation plot']

    accuracies = dict()
    overlaps = dict()
    with open(evaluation_path) as f:
        f.readline()
        for r in map(lambda l: l.strip().split(','), f):
            accuracies[r[0]] = float(r[3])
            overlaps[r[0]] = float(r[4])
    logging.log(level=logging.INFO, msg='loaded evaluation results')

    plot_accuracy_overlap_plot(
        'comparison of ts+distknn with different k',
        'k neighbors',
        k_neighbors,
        np.array([accuracies['ts+distknn(k={})'.format(k)] for k in k_neighbors]),
        np.array([overlaps['ts+distknn(k={})'.format(k)] for k in k_neighbors]),
        ts_knn_plot_output
    )

    plot_accuracy_overlap_plot(
        'comparison of ts+linproj with different c',
        'c clusters',
        c_clusters,
        np.array([accuracies['ts+linproj(c={})'.format(c)] for c in c_clusters]),
        np.array([overlaps['ts+linproj(c={})'.format(c)] for c in c_clusters]),
        ts_linproj_plot_output
    )


if __name__ == '__main__':
    main()
