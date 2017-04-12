import json
import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    offsets_path = paths_config['subclass offsets']
    plot_output = paths_config['subclass offset plot']
    max_offset_amount = int(8e5)  # type: int

    offsets = list()
    c = 0

    with open(offsets_path) as f:
        for subclass, superclass, offset in map(lambda s: s.strip().split(';'), f):
            # i don't know why i would do this
            offset = np.array(offset.strip('[').strip(']').strip().split(), dtype=np.float32)
            offsets.append(offset)
            c += 1
    logging.log(level=logging.INFO, msg='total of {} offsets loaded'.format(c))

    offsets = np.array(offsets)
    if max_offset_amount != -1:
        offsets = np.random.permutation(offsets)[:max_offset_amount]
        logging.log(level=logging.INFO, msg='random sampled {} offsets'.format(offsets.shape[0]))

    pca = PCA(n_components=2)
    pca.fit(offsets)
    offsets = pca.transform(offsets)
    logging.log(level=logging.INFO, msg='executed pca')

    ms = KMeans(n_clusters=9, n_jobs=3)
    clusters = ms.fit_predict(offsets)
    logging.log(level=logging.INFO, msg='executed clustering on 2d offsets')

    xs, ys = offsets.T

    colors = ['#FF4405', '#FFC105', '#C1FF05', '#44FF05', '#FF9F80', '#05FF44', '#FF05C1', '#42D0FF', '#4405FF']

    logging.log(level=logging.INFO, msg='create plot with {} points'.format(c))
    plt.scatter(xs, ys, s=0.1, edgecolors=[colors[cluster] for cluster in clusters])
    logging.log(level=logging.INFO, msg='save plot to {}'.format(plot_output))
    plt.savefig(plot_output)

if __name__ == '__main__':
    main()
