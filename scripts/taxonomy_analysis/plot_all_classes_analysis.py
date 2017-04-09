import json
import logging

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Set


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    analysis_paths = [
        paths_config['class analysis'],
        paths_config['orphan class analysis']
    ]

    analysis = dict()
    for analysis_path in analysis_paths:
        with open(analysis_path) as f:
            analysis[analysis_path] = json.load(f)
    logging.log(level=logging.INFO, msg='loaded analysis')

    labels_path = paths_config['wikidata labels']

    shorten_labels = 24

    frequency_limit = 25
    frequency_plots = [
        {'title': 'most common properties in all classes',
         'analysis': paths_config['class analysis'],
         'out path': paths_config['class property frequency'],
         'key': 'property histogram'},

        {'title': 'most common properties in orphan classes',
         'analysis': paths_config['orphan class analysis'],
         'out path': paths_config['orphan class property frequency'],
         'key': 'property histogram'},
    ]

    count_limit = 15
    count_plots = [
        {'title': 'property count histogram for all classes',
         'x': 'number of properties',
         'analysis': paths_config['class analysis'],
         'out path': paths_config['class property counts'],
         'key': 'property count histogram'},
        {'title': 'subclass count histogram for all classes',
         'x': 'number of subclasses',
         'analysis': paths_config['class analysis'],
         'out path': paths_config['class subclass counts'],
         'key': 'subclass count histogram'},
        {'title': 'instance count histogram for all classes',
         'x': 'number of instances',
         'analysis': paths_config['class analysis'],
         'out path': paths_config['class instance counts'],
         'key': 'instance count histogram'},

        {'title': 'property count histogram for orphan classes',
         'x': 'number of properties',
         'analysis': paths_config['orphan class analysis'],
         'out path': paths_config['orphan class property counts'],
         'key': 'property count histogram'},
        {'title': 'subclass count histogram for orphan classes',
         'x': 'number of subclasses',
         'analysis': paths_config['orphan class analysis'],
         'out path': paths_config['orphan class subclass counts'],
         'key': 'subclass count histogram'},
        {'title': 'instance count histogram for orphan classes',
         'x': 'number of instances',
         'analysis': paths_config['orphan class analysis'],
         'out path': paths_config['orphan class instance counts'],
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
                        sorted(analysis[frequency_plot['analysis']][frequency_plot['key']].items(),
                               key=lambda i: int(i[1]))))
        top_ids = list(x[::-1][:frequency_limit])  # type: List[str]
        # normalized values
        top_values = np.array((y[::-1][:frequency_limit])) / analysis[frequency_plot['analysis']]['class count']

        if not set(top_ids).issubset(set(labels.keys())):
            add_labels(set(top_ids).difference(set(labels.keys())))

        # labels have the following format: "<english label> (<pid>)",
        # if english label is longer than shorten_labels, the label is cut off at shorten_labels-3 and ... is appended.
        top_labels = ['{} ({})'.format(
            labels[tid] if len(labels[tid]) <= shorten_labels else labels[tid][:shorten_labels-3] + '...',
            tid) for tid in top_ids]   # type: List[str]

        ax.barh(np.arange(frequency_limit), top_values, color='#3f9b0b')
        ax.set_title(frequency_plot['title'])

        ax.set_xlabel('percentage of classes')
        ax.set_yticks(np.arange(frequency_limit))
        ax.set_yticklabels(top_labels, va='bottom')
        plt.tight_layout()

        plt.savefig(frequency_plot['out path'])
        logging.log(logging.INFO, msg='stored frequency plot to {}'.format(frequency_plot['out path']))

    for count_plot in count_plots:
        plt.figure(1)
        plt.clf()

        x, y = zip(*map(lambda t: (int(t[0]), int(t[1])),
                        sorted(analysis[count_plot['analysis']][count_plot['key']].items(), key=lambda i: int(i[0]))))
        top_x = np.array(x[:count_limit])
        top_y = np.array(y[:count_limit]) / np.sum(np.array(y))

        plt.xlim([-0.5, count_limit])
        plt.bar(top_x, top_y, color='#2c6fbb', align='center')
        plt.xlabel(count_plot['x'])
        plt.ylabel('number of classes')
        plt.title(count_plot['title'])

        plt.savefig(count_plot['out path'])
        logging.log(logging.INFO, msg='stored count plot to {}'.format(count_plot['out path']))

if __name__ == '__main__':
    main()
