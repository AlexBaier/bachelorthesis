import logging
import time

from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from data_analysis.pipe import DataPipe
import data_analysis.utils as utils


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    embeddings_path = '../data/algorithm_io/simple_sentence_class_embeddings-20161107'
    classes_path = '../data/classes-20161107'
    plot_path = '../data/plots/2d_class_embeddings.png'
    plot_count = 100

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

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    embeddings = tsne.fit_transform(embeddings[:min(plot_count, embeddings.shape[0])])

    del tsne

    duration = time.time() - start_time
    logging.log(level=logging.INFO, msg='executed tsne in {} seconds'.format(duration))

    plt.figure(figsize=(18, 18))
    xs, ys = embeddings.T
    plt.scatter(xs, ys)
    texts = list()
    for idx, embedding in enumerate(embeddings):
        x, y = embedding
        class_id = class_ids[idx]
        label = class_labels[class_id] if class_labels[class_id] else class_id
        texts.append(plt.text(x, y, label))
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    logging.log(level=logging.INFO, msg='finished drawing scatter plot')

    plt.savefig(plot_path)
    logging.log(logging.INFO, msg='saved scatter plot to {}'.format(plot_path))


if __name__ == '__main__':
    main()
