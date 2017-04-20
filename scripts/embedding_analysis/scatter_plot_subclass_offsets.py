import json
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# copied from https://gist.github.com/adewes/5884820
def get_random_color(pastel_factor=0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0, 1.0) for _ in [1, 2, 3]]]


# copied from https://gist.github.com/adewes/5884820
def color_distance(c1, c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1, c2)])


# copied from https://gist.github.com/adewes/5884820
def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    offsets_path = paths_config['subclass offsets']
    plot_output = paths_config['subclass offset plot']
    max_offset_amount = int(8e5)  # type: int

    offsets = list()
    n_clusters = 10

    c_offset = 0
    with open(offsets_path) as f:
        for subclass, superclass, offset in map(lambda s: s.strip().split(';'), f):
            # i don't know why i would do this
            offset = np.array(offset.strip('[').strip(']').strip().split(), dtype=np.float32)
            offsets.append(offset)
            c_offset += 1
    logging.log(level=logging.INFO, msg='total of {} offsets loaded'.format(c_offset))

    offsets = np.array(offsets)
    if max_offset_amount != -1:
        offsets = np.random.permutation(offsets)[:max_offset_amount]
        logging.log(level=logging.INFO, msg='random sampled {} offsets'.format(offsets.shape[0]))

    ms = KMeans(n_clusters=n_clusters, n_jobs=3)
    clusters = ms.fit_predict(offsets)
    logging.log(level=logging.INFO, msg='executed clustering on offsets')

    pca = PCA(n_components=2)
    pca.fit(offsets)
    offsets = pca.transform(offsets)
    cluster_centers = pca.transform(ms.cluster_centers_)
    logging.log(level=logging.INFO, msg='executed pca')

    xs, ys = offsets.T
    c_xs, c_ys = cluster_centers.T

    colors = []
    for i in range(n_clusters):
        colors.append(generate_new_color(colors, pastel_factor=0))

    logging.log(level=logging.INFO, msg='create plot with {} points'.format(c_offset))
    plt.scatter(xs, ys, s=0.1, edgecolors=[colors[cluster] for cluster in clusters])
    for c in range(n_clusters):
        plt.text(c_xs[c], c_ys[c], str(c))
    logging.log(level=logging.INFO, msg='save plot to {}'.format(plot_output))
    plt.savefig(plot_output)

if __name__ == '__main__':
    main()
