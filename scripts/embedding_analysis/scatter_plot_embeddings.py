import json
import logging

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from evaluation.utils import load_embeddings_and_labels


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('algorithm_config.json') as f:
        algorithm_config = json.load(f)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    embeddings_path = algorithm_config['triple sentence sgns']['embeddings path']
    plot_output = paths_config['embeddings plot']

    embeddings, _ = load_embeddings_and_labels(embeddings_path)
    logging.log(level=logging.INFO, msg='loaded embeddings')

    pca = PCA(n_components=2)
    pca.fit(embeddings)
    embeddings = pca.transform(embeddings)
    logging.log(level=logging.INFO, msg='executed pca')

    ms = KMeans(n_clusters=9, n_jobs=3)
    clusters = ms.fit_predict(embeddings)
    logging.log(level=logging.INFO, msg='executed clustering on 2d offsets')

    xs, ys = embeddings.T

    colors = ['#FF4405', '#FFC105', '#C1FF05', '#44FF05', '#FF9F80', '#05FF44', '#FF05C1', '#42D0FF', '#4405FF']

    plt.scatter(xs, ys, s=0.1, edgecolors=[colors[cluster] for cluster in clusters])
    logging.log(level=logging.INFO, msg='save plot to {}'.format(plot_output))
    plt.savefig(plot_output)

if __name__ == '__main__':
    main()
