import json
import logging

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from evaluation.utils import load_embeddings_and_labels, generate_new_color


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('algorithm_config.json') as f:
        algorithm_config = json.load(f)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    embeddings_path = algorithm_config['triple sentence sgns']['embeddings path']
    plot_output = paths_config['embeddings plot']

    n_clusters = 50

    embeddings, _ = load_embeddings_and_labels(embeddings_path)
    logging.log(level=logging.INFO, msg='loaded embeddings')

    ms = KMeans(n_clusters=n_clusters, n_jobs=3)
    clusters = ms.fit_predict(embeddings)
    logging.log(level=logging.INFO, msg='executed clustering on embeddings')

    pca = PCA(n_components=2)
    pca.fit(embeddings)
    embeddings = pca.transform(embeddings)
    logging.log(level=logging.INFO, msg='executed pca')

    xs, ys = embeddings.T

    colors = []
    for i in range(n_clusters):
        colors.append(generate_new_color(colors, pastel_factor=0))

    plt.scatter(xs, ys, s=0.1, color=[colors[cluster] for cluster in clusters])
    logging.log(level=logging.INFO, msg='save plot to {}'.format(plot_output))
    plt.savefig(plot_output)

if __name__ == '__main__':
    main()
