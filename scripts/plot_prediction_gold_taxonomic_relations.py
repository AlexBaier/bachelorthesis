import logging

import matplotlib.pyplot as plt
import numpy as np


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    eval_result_path = '../evaluation/hybrid_evaluation-20161107.csv'

    output_path = '../evaluation/plots/tax_rel_{}-20161107.png'

    combined_plots = [
        ['ts+linproj', 'ts+pwlinproj(c=30)', 'ts+kriknn(k=15&r=1)']
    ]

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
            distance_exceeded[algorithm] = (fps[algorithm] - overspecialized[algorithm] - underspecialized[algorithm]
                                            - common_parent[algorithm])
            logging.log(level=logging.INFO, msg='loaded {}'.format(algorithm))

    colors = [
        '#929591',  # grey
        '#3f9b0b',  # grass green
        '#8f1402',  # brick red
        '#75bbfd',  # sky blue
    ]

    # plot ts+kriknn(k=?&r=1) with different k's and r=1
    plt.figure(1)
    plt.clf()

    algorithm = 'ts+kriknn(k={}&r=1)'
    ks = [2, 5, 10, 15]

    n = len(ks)
    ind = np.arange(n)

    de_c = np.array([distance_exceeded[algorithm.format(k)] for k in ks])
    os_c = np.array([overspecialized[algorithm.format(k)] for k in ks])
    us_c = np.array([underspecialized[algorithm.format(k)] for k in ks])
    cp_c = np.array([common_parent[algorithm.format(k)] for k in ks])

    de_plt = plt.bar(ind, de_c, color=colors[0])
    os_plt = plt.bar(ind, os_c, color=colors[1], bottom=de_c)
    us_plt = plt.bar(ind, us_c, color=colors[2], bottom=de_c + os_c)
    cp_plt = plt.bar(ind, cp_c, color=colors[3], bottom=de_c + os_c + us_c)

    plt.title('ts+kriknn(k=?&r=1): taxonomic relations of misclassifications')
    plt.xlabel('k')
    plt.ylabel('count')
    plt.legend([de_plt[0], os_plt[0], us_plt[0], cp_plt[0]],
               ['distance exceeded', 'overspecialized', 'underspecialized', 'same parent'], loc='lower left')
    plt.xticks(ind + 0.4, ks)

    plt.savefig(output_path.format('_'.join([algorithm.format(k) for k in ks])))

    # plot arbitrary algorithm combinations defined in combined_plots
    for combined_plot in combined_plots:
        plt.figure(1)
        plt.clf()

        n = len(combined_plot)
        ind = np.arange(n)

        de_c = np.array([distance_exceeded[algorithm] for algorithm in combined_plot])
        os_c = np.array([overspecialized[algorithm] for algorithm in combined_plot])
        us_c = np.array([underspecialized[algorithm] for algorithm in combined_plot])
        cp_c = np.array([common_parent[algorithm] for algorithm in combined_plot])

        de_plt = plt.bar(ind, de_c, color=colors[0])
        os_plt = plt.bar(ind, os_c, color=colors[1], bottom=de_c)
        us_plt = plt.bar(ind, us_c, color=colors[2], bottom=de_c+os_c)
        cp_plt = plt.bar(ind, cp_c, color=colors[3], bottom=de_c+os_c+us_c)

        plt.title('taxonomic relations of misclassifications')
        plt.xlabel('algorithm')
        plt.ylabel('count')
        plt.legend([de_plt[0], os_plt[0], us_plt[0], cp_plt[0]],
                   ['distance exceeded', 'overspecialized', 'underspecialized', 'same parent'], loc='lower left')
        plt.xticks(ind+0.4, combined_plot)

        plt.savefig(output_path.format('_'.join(combined_plot)))


if __name__ == '__main__':
    main()
