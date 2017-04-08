import json
import logging

import matplotlib.pyplot as plt


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    with open(paths_config['wikidata labels']) as f:
        labels = json.load(f)
    logging.log(level=logging.INFO, msg='loaded labels')

    with open(paths_config['class analysis']) as f:
        analysis = json.load(f)
    logging.log(level=logging.INFO, msg='loaded {}'.format(analysis))

    frequency_limit = 20
    frequency_plots = [
        {'title': 'property frequencies of all classes',
         'path': paths_config['class property frequency'],
         'key': 'property histogram'},
        {'title': 'subclass frequencies of all classes',
         'path': paths_config['class subclass frequency'],
         'key': 'subclass histogram'}
    ]

    count_limit = 40
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

    for frequency_plot in frequency_plots:
        plt.figure(1)
        plt.clf()

        x, y = map(lambda t: (t[0], int(t[1])), sorted(analysis[frequency_plot['key']].items(), key=lambda i: i[1]))
        top_labels = [labels[pid] for pid in x[::-1][:frequency_limit]]
        top_values = y[::-1][:frequency_limit]
        rest_sum = sum(y[::-1][frequency_limit:])

        labels = top_labels + ['other properties']
        values = top_values + [rest_sum]

        plt.pie(values, labels=labels, startangle=90)
        plt.axis('equal')
        plt.title(frequency_plot['title'])
        plt.legend()

        plt.savefig(frequency_plot['path'])
        logging.log(logging.INFO, msg='stored frequency plot to {}'.format(frequency_plot['path']))

    for count_plot in count_plots:
        plt.figure(1)
        plt.clf()

        x, y = zip(*sorted(map(lambda t: (int(t[0], int(1))), analysis[count_plot['key']]), key=lambda i: i[0]))
        top_x = x[:count_limit]
        top_y = y[:count_limit]

        plt.bar(top_x, top_y, color='#380282')  # color: indigo
        plt.xlabel(count_plot['x'])
        plt.ylabel('number of classes')
        plt.title(count_plot['title'])

        plt.savefig(count_plot['path'])
        logging.log(logging.INFO, msg='stored count plot to {}'.format(count_plot['path']))

if __name__ == '__main__':
    main()
