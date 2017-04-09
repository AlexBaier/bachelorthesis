import json
import logging

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Set


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    with open(paths_config['class analysis']) as f:
        analysis = json.load(f)
    logging.log(level=logging.INFO, msg='loaded analysis')

    labels_path = paths_config['wikidata labels']

    frequency_limit = 25
    frequency_plots = [
        {'title': 'property frequencies of all classes',
         'path': paths_config['class property frequency'],
         'key': 'property histogram'}
    ]

    count_limit = 15
    count_plots = [
        {'title': 'property count histogram for all classes',
         'x': 'number of properties',
         'path': paths_config['class property counts'],
         'key': 'property count histogram'},
        {'title': 'subclass count histogram for all classes',
         'x': 'number of subclasses',
         'path': paths_config['class subclass counts'],
         'key': 'subclass count histogram'},
        {'title': 'instance count histogram for all classes',
         'x': 'number of instances',
         'path': paths_config['class instance counts'],
         'key': 'instance count histogram'}
    ]

    labels = dict()  # type: Dict[str, str]

    def add_labels(cids: Set[str]):
        with open(labels_path) as g:
            for eid, label in map(lambda l: l.strip().split(',', 1), g):
                if eid in cids:
                    labels[eid] = label

    for frequency_plot in frequency_plots:
        plt.figure(1)
        plt.clf()

        fig, ax = plt.subplots()

        x, y = zip(*map(lambda t: (t[0], int(t[1])),
                        sorted(analysis[frequency_plot['key']].items(), key=lambda i: int(i[1]))))
        top_ids = list(x[::-1][:frequency_limit])  # type: List[str]
        # normalized values
        top_values = np.array((y[::-1][:frequency_limit])) / np.sum(np.array(y))

        if not set(top_ids).issubset(set(labels.keys())):
            add_labels(set(top_ids).difference(set(labels.keys())))

        top_labels = ['{} ({})'.format(labels[tid], tid) for tid in top_ids]   # type: List[str]

        top_labels = top_labels + ['other properties']  # type: List[str]

        ax.barh(np.arange(frequency_limit), top_values, color='#3f9b0b')
        ax.set_title(frequency_plot['title'])

        ax.set_yticks(np.arange(frequency_limit))
        ax.set_yticklabels(top_labels, va='bottom')
        plt.tight_layout()

        plt.savefig(frequency_plot['path'])
        logging.log(logging.INFO, msg='stored frequency plot to {}'.format(frequency_plot['path']))

    for count_plot in count_plots:
        plt.figure(1)
        plt.clf()

        x, y = zip(*map(lambda t: (int(t[0]), int(t[1])),
                        sorted(analysis[count_plot['key']].items(), key=lambda i: int(i[0]))))
        top_x = np.array(x[:count_limit])
        top_y = np.array(y[:count_limit]) / np.sum(np.array(y))

        plt.xlim([-0.5, count_limit])
        plt.bar(top_x, top_y, color='#2c6fbb', align='center')
        plt.xlabel(count_plot['x'])
        plt.ylabel('number of classes')
        plt.title(count_plot['title'])

        plt.savefig(count_plot['path'])
        logging.log(logging.INFO, msg='stored count plot to {}'.format(count_plot['path']))

if __name__ == '__main__':
    main()
