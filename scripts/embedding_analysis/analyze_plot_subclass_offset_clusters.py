import json
import logging
import random
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from evaluation.utils import generate_new_color


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    offsets_path = paths_config['subclass offsets']
    labels_path = paths_config['wikidata labels']
    plot_output = paths_config['subclass offset plot']
    result_path = paths_config['subclass offset cluster']

    n_clusters = 15
    max_labeled_pairs = 50

    offsets = list()
    ids = list()
    c_offset = 0
    result = dict()
    result['plot'] = plot_output

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
    logging.log(level=logging.INFO, msg='executed pca')

    xs, ys = offsets.T

    colors = []
    for i in range(n_clusters):
        colors.append(generate_new_color(colors, pastel_factor=0))

    result['colors'] = colors

    logging.log(level=logging.INFO, msg='create plot with {} points'.format(c_offset))
    handles = list()
    for cluster in range(n_clusters):
        handle = plt.scatter([x[1] for x in filter(lambda v: clusters[v[0]] == cluster, enumerate(xs))],
                             [y[1] for y in filter(lambda v: clusters[v[0]] == cluster, enumerate(ys))],
                             s=0.1, color=colors[cluster])
        handles.append(handle)
    legend = plt.legend(handles, list(range(n_clusters)), scatterpoints=1)
    for handle in legend.legendHandles:
        handle._sizes = [30]
    logging.log(level=logging.INFO, msg='save plot to {}'.format(plot_output))
    plt.savefig(plot_output)

    # get number of pairs in each cluster
    result['counts'] = [len(list(group)) for key, group in groupby(sorted(clusters.tolist()))]

    # ignore the biggest cluster to improve label load time
    result['pairs'] = dict()
    for idx, cluster in enumerate(clusters):
        if not result['pairs'].get(str(cluster), None):
            result['pairs'][str(cluster)] = list()
        result['pairs'][str(cluster)].append({
            'subclass': ids[idx][0],
            'superclass': ids[idx][1],
            })

    # shuffle pairs and cut off at limit and collect ids which need to be labeled
    to_label = set()
    for cluster in range(n_clusters):
        random.shuffle(result['pairs'][str(cluster)])
        result['pairs'][str(cluster)] = result['pairs'][str(cluster)][:max_labeled_pairs]
        for pair in result['pairs'][str(cluster)]:
            to_label.add(pair['subclass'])
            to_label.add(pair['superclass'])
    logging.log(level=logging.INFO, msg='labels required: {}'.format(len(to_label)))

    labels = dict()
    with open(labels_path) as g:
        for eid, label in map(lambda l: l.strip().split(',', 1), g):
            if eid in to_label:
                labels[eid] = label
    logging.log(level=logging.INFO, msg='loaded labels: {}/{}'.format(len(labels.items()), len(to_label)))

    # add labels to results
    for cluster in range(n_clusters):
        for pair in result['pairs'][str(cluster)]:
            pair['subclass label'] = labels.get(pair['subclass'], ''),
            pair['superclass label'] = labels.get(pair['superclass'], '')

    with open(result_path, mode='w') as f:
        json.dump(result, f)
    logging.log(level=logging.INFO, msg='wrote subclass-superclass clusters to {}'.format(result_path))

if __name__ == '__main__':
    main()
