import json
import logging
import random
from itertools import groupby

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
    labels_path = paths_config['wikidata labels']
    plot_output = paths_config['subclass offset plot']
    result_path = paths_config['subclass offset cluster']

    n_clusters = 10
    max_labeled_pairs = 50

    offsets = list()
    ids = list()
    c_offset = 0

    with open(offsets_path) as f:
        for subclass, superclass, offset in map(lambda s: s.strip().split(';'), f):
            # i don't know why i would do this
            offset = np.array(offset.strip('[').strip(']').strip().split(), dtype=np.float32)
            offsets.append(offset)
            ids.append((subclass, superclass))
            c_offset += 1
    logging.log(level=logging.INFO, msg='total of {} offsets loaded'.format(c_offset))

    offsets = np.array(offsets)

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
        plt.text(c_xs[c], c_ys[c], str(c), fontsize='smaller')
    logging.log(level=logging.INFO, msg='save plot to {}'.format(plot_output))
    plt.savefig(plot_output)

    result = dict()

    # get number of pairs in each cluster
    result['counts'] = [len(list(group)) for key, group in groupby(sorted(clusters.tolist()))]

    # ignore the biggest cluster to improve label load time
    ignore = np.argmax(result['counts'])
    to_label = set()
    for idx, cluster in enumerate(clusters):
        if cluster != ignore:
            to_label.update(ids[idx])
    logging.log(level=logging.INFO, msg='labels required: {}'.format(len(to_label)))

    labels = dict()
    with open(labels_path) as g:
        for eid, label in map(lambda l: l.strip().split(',', 1), g):
            if eid in to_label:
                labels[eid] = label
    logging.log(level=logging.INFO, msg='loaded labels: {}/{}'.format(len(labels.items()), len(to_label)))

    result['pairs'] = dict()
    for idx, cluster in enumerate(clusters):
        if cluster != ignore:
            if not result['pairs'].get(str(cluster), None):
                result['pairs'][str(cluster)] = list()
            result['pairs'][str(cluster)].append({
                'subclass': ids[idx][0],
                'superclass': ids[idx][1],
                'subclass label': labels.get(ids[idx][0], ''),
                'superclass label': labels.get(ids[idx][1], '')
            })

    # shuffle pairs and cut off at limit
    for cluster in range(n_clusters):
        if cluster != ignore:
            random.shuffle(result['pairs'][str(cluster)])
            result['pairs'][str(cluster)] = result['pairs'][str(cluster)][:max_labeled_pairs]

    with open(result_path, mode='w') as f:
        json.dump(result, f)
    logging.log(level=logging.INFO, msg='wrote subclass-superclass clusters to {}'.format(result_path))

if __name__ == '__main__':
    main()
