import json
import logging

import matplotlib.pyplot as plt
import numpy as np

from evaluation.utils import algo2color


def plot_overlap_histogram(title, bins, overlaps, colors, legend_labels, output_path):
    plt.figure(1)
    plt.clf()
    data = np.vstack(overlaps).T
    _, bins, _ = plt.hist(data, bins, color=colors, label=legend_labels)
    plt.xticks(bins)
    plt.title(title)
    plt.xlabel('local taxonomic overlap')
    plt.ylabel('percentage of prediction-gold pairs in bin')
    plt.legend(loc='upper center')
    plt.savefig(output_path)
    logging.log(level=logging.INFO, msg='stored {} to {}'.format(title, output_path))


def plot_overlap_distribution(title, bins, overlaps, colors, legend_labels, output_path):
    plt.figure(2)
    plt.clf()
    plt.ylim([0.0, 1.0])
    data = np.vstack(overlaps).T
    _, bins, _ = plt.hist(data, bins, color=colors, label=legend_labels, cumulative=True, normed=True)
    plt.xticks(bins)
    plt.title(title)
    plt.xlabel('taxonomic overlap x')
    plt.ylabel('$P[x \leq X]$')
    plt.legend(loc='upper center')
    plt.savefig(output_path)
    logging.log(level=logging.INFO, msg='stored {} to {}'.format(title, output_path))


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    algorithms = ['ts+distknn(k=5)', 'ts+distknn(k=10)', 'ts+distknn(k=15)', 'ts+distknn(k=20)',
                  'ts+linproj(c=1)', 'ts+linproj(c=25)', 'ts+linproj(c=50)',
                  'gw+distknn(k=5)', 'gw+distknn(k=10)', 'gw+distknn(k=15)', 'gw+distknn(k=20)',
                  'gw+linproj(c=1)', 'gw+linproj(c=25)', 'gw+linproj(c=50)']
    n_bins = 10

    k_neighbors = [5, 10, 15, 20]
    c_clusters = [1, 25, 50]

    with open('paths_config.json') as f:
        config = json.load(f)

    overlaps_path = config['local taxonomic overlaps']
    knn_hist_path = config['ts+distknn overlap hist']
    knn_distr_path = config['ts+distknn overlap distribution']
    linproj_hist_path = config['ts+linproj overlap hist']
    linproj_distr_path = config['ts+linproj overlap distribution']

    overlaps = dict()
    for algorithm in algorithms:
        with open(overlaps_path.format(algorithm)) as f:
            overlaps[algorithm] = list(map(float, f.readline().strip().split(',')))

    plot_overlap_histogram(
        'histogram of local taxonomic overlaps for ts+distknn',
        n_bins,
        [overlaps['ts+distknn(k={})'.format(k)] for k in k_neighbors],
        [algo2color('ts+distknn(k={})'.format(k)) for k in k_neighbors],
        ['k = {}'.format(k) for k in k_neighbors],
        knn_hist_path
    )

    plot_overlap_histogram(
        'histogram of local taxonomic overlaps for ts+linproj',
        n_bins,
        [overlaps['ts+linproj(c={})'.format(c)] for c in c_clusters],
        [algo2color('ts+linproj(c={})'.format(c)) for c in c_clusters],
        ['c = {}'.format(c) for c in c_clusters],
        linproj_hist_path
    )

    plot_overlap_distribution(
        'distribution of local taxonomic overlaps for ts+distknn',
        n_bins,
        [overlaps['ts+distknn(k={})'.format(k)] for k in k_neighbors],
        [algo2color('ts+distknn(k={})'.format(k)) for k in k_neighbors],
        ['k = {}'.format(k) for k in k_neighbors],
        knn_distr_path
    )

    plot_overlap_distribution(
        'distribution of local taxonomic overlaps for ts+linproj',
        n_bins,
        [overlaps['ts+linproj(c={})'.format(c)] for c in c_clusters],
        [algo2color('ts+linproj(c={})'.format(c)) for c in c_clusters],
        ['c = {}'.format(c) for c in c_clusters],
        linproj_distr_path
    )


if __name__ == '__main__':
    main()
