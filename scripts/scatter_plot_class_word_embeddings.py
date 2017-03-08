import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import IncrementalPCA

from data_analysis.pipe import DataPipe
import data_analysis.utils as utils


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    embeddings_path = '../data/algorithm_io/simple_sentence_class_embeddings-20161107'
    classes_path = '../data/classes-20161107'
    plot_path = '../data/plots/2d_class_embeddings.png'
    plot_count = 500

    embeddings = list()
    class_ids = list()
    class_labels = dict(DataPipe.read_json_dump(classes_path)
                        .map(lambda c: (c['id'], utils.get_english_label(c))).to_list())
    logging.log(level=logging.INFO, msg='loaded {} class labels'.format(len(list(class_labels.keys()))))

    with open(embeddings_path) as f:
        for cid, embedding in map(lambda l: l.strip().split(';'), f):
            embedding = np.array(embedding.strip('[').strip(']').strip().split(), dtype=np.float32)
            embeddings.append(embedding)
            class_ids.append(cid)

    embeddings = np.array(embeddings)
    logging.log(level=logging.INFO, msg='loaded {} class word embeddings'.format(len(class_ids)))

    start_time = time.time()

    pca = IncrementalPCA(n_components=2)
    pca.fit(embeddings)
    embeddings = pca.transform(embeddings)
    xs, ys = embeddings.T

    del pca

    duration = time.time() - start_time
    logging.log(level=logging.INFO, msg='executed PCA in {} seconds'.format(duration))

    plt.scatter(xs, ys)
    for i, embedding in enumerate(embeddings[:plot_count]):
        x, y = embedding
        label = class_labels[class_ids[i]] if class_labels[class_ids[i]] else class_ids[i]
        plt.annotate(label, xy=(x, y))
    logging.log(level=logging.INFO, msg='finished drawing scatter plot')

    plt.savefig(plot_path)
    logging.log(logging.INFO, msg='saved scatter plot to {}'.format(plot_path))


if __name__ == '__main__':
    main()
