import logging

import matplotlib.pyplot as plt


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    eval_result_path = '../evaluation/hybrid_evaluation-20161107.csv'

    output_path = '../evaluation/plots/tax_rel_pie_{}-20161107.png'

    algorithms = list()
    totals = dict()
    tps = dict()
    fps = dict()
    overspecialized = dict()
    underspecialized = dict()
    common_parent = dict()
    distance_exceeded = dict()

    with open(eval_result_path) as f:
        label2idx = dict((label, idx) for idx, label in enumerate(f.readline().split(',')))
        for row in list(map(lambda r: r.split(','), f.readlines())):
            algorithm = row[label2idx['algorithm']]
            algorithms.append(algorithm)
            totals[algorithm] = int(row[label2idx['total']])
            tps[algorithm] = int(row[label2idx['TPs']])
            fps[algorithm] = totals[algorithm] - tps[algorithm]
            overspecialized[algorithm] = int(row[label2idx['overspecialized']])
            underspecialized[algorithm] = int(row[label2idx['underspecialized']])
            common_parent[algorithm] = int(row[label2idx['same_parent']])
            distance_exceeded[algorithm] = fps[algorithm] - overspecialized[algorithm] - underspecialized[algorithm] \
                                           - common_parent[algorithm]
            logging.log(level=logging.INFO, msg='loaded {}'.format(algorithm))

    labels = ['underspecialized', 'overspecialized', 'same parent']
    colors = [
        '#06470c',  # forest green
        '#8f1402',  # brick red
        '#75bbfd',  # sky blue
    ]
    for algorithm in algorithms:
        plt.figure(1)
        plt.clf()
        values = [underspecialized[algorithm],
                  overspecialized[algorithm],
                  common_parent[algorithm]]
        total = sum(values)
        patches, texts, autotexts = plt.pie(values, startangle=90, colors=colors, pctdistance=0.5,
                                            autopct=lambda p: '{:.0f}'.format(p * total / 100))
        plt.axis('equal')
        plt.legend(patches, labels, loc='lower left')
        plt.text(0.55, 0.85, 'distance exceeded: {}'.format(distance_exceeded[algorithm]))
        plt.title('{}: taxonomic relations of misclassifications'.format(algorithm))

        plt.savefig(output_path.format(algorithm))
        logging.log(level=logging.INFO, msg='stored plot to {}'.format(output_path.format(algorithm)))


if __name__ == '__main__':
    main()
