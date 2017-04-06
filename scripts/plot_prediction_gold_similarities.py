import json
import logging

import matplotlib.pyplot as plt
import numpy as np

from evaluation.statistics import get_prediction_gold_cosine_similarities
from evaluation.utils import algo2color, load_embeddings_and_labels, load_test_data


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    config_path = '../algorithm_config.json'
    predictions_path = '../evaluation/results_{}-20161107.csv'
    test_data_path = '../evaluation/test_data-20161107.csv'

    sim_hist_plot_path = '../evaluation/plots/sim_hist_{}-20161107.png'
    leq_sim_distr_plot_path = '../evaluation/plots/cumul_sim_distr_{}-20161107.png'

    comb_sim_hist_plot_path = '../evaluation/plots/comb_sim_hist_{}-20161107.png'
    comb_leq_sim_distr_plot_path = '../evaluation/plots/comb_cumul_sim_distr_{}-20161107.png'

    algorithms = [
        'ts+kriknn(k=2&r=1)',
        'ts+kriknn(k=5&r=1)',
        'ts+kriknn(k=10&r=1)',
        'ts+kriknn(k=15&r=1)',
        'ts+kriknn(k=5&r=10)',
        'ts+kriknn(k=15&r=10)',
        'ts+distknn(k=15)',
        'ts+linproj',
        'ts+pwlinproj(c=30)',
        'ts+pwlinproj(c=50)',
        'ts+pwlinproj(c=70)',
        'ts+pwlinproj(c=85)',
        'ts+pwlinproj(c=100)'
    ]
    combined_plots = [
        ['ts+kriknn(k=15&r=1)', 'ts+distknn(k=15)'],
        ['ts+kriknn(k=2&r=1)', 'ts+kriknn(k=5&r=1)', 'ts+kriknn(k=10&r=1)', 'ts+kriknn(k=15&r=1)'],
        ['ts+kriknn(k=5&r=10)', 'ts+kriknn(k=15&r=10)'],
        ['ts+kriknn(k=15&r=1)', 'ts+kriknn(k=15&r=10)'],
        ['ts+kriknn(k=5&r=1)', 'ts+kriknn(k=5&r=10)', 'ts+kriknn(k=15&r=1)', 'ts+kriknn(k=15&r=10)'],
        ['ts+linproj', 'ts+pwlinproj(c=30)', 'ts+pwlinproj(c=50)', 'ts+pwlinproj(c=70)', 'ts+pwlinproj(c=85)'],
        ['ts+kriknn(k=15&r=1)', 'ts+pwlinproj(c=50)']
    ]

    nbins = 12
    round_to = 3

    with open(config_path) as f:
        config = json.load(f)
    logging.log(level=logging.INFO, msg='loaded algorithm config')

    golds = load_test_data(test_data_path)
    logging.log(level=logging.INFO, msg='loaded gold standard')

    predictions = dict()
    for algorithm in algorithms:
        with open(predictions_path.format(algorithm)) as f:
            predictions[algorithm] = dict((u, p) for u, p in map(lambda l: l.strip().split(','), f))
        logging.log(level=logging.INFO, msg='loaded predictions of {}'.format(algorithm))

    id2embedding = dict()
    for algorithm in algorithms:
        model = config['combinations'][algorithm]['sgns']
        if model in id2embedding.keys():
            continue
        embeddings, labels = load_embeddings_and_labels(config[model]['embeddings path'])
        id2idx = dict((label, idx) for idx, label in enumerate(labels))
        id2embedding[model] = lambda item_id: embeddings[id2idx[item_id]]
        logging.log(level=logging.INFO, msg='loaded embeddings of {}'.format(model))

    similarities = dict(
        (algorithm,
         np.round(get_prediction_gold_cosine_similarities(
             predictions[algorithm], golds,
             id2embedding[config['combinations'][algorithm]['sgns']]), decimals=round_to))
        for algorithm in algorithms)
    logging.log(level=logging.INFO, msg='computed similarities between all predictions and gold standards')

    for algorithm in algorithms:
        # plot similarity histogram
        plt.figure(1)
        plt.clf()
        axes = plt.gca()
        n, bins, _ = plt.hist(similarities[algorithm], bins=nbins, label=algorithm, color=algo2color(algorithm))
        for idx, value in enumerate(n):
            plt.text(bins[idx], value+1, s=str(int(value)))
        axes.relim()
        axes.autoscale()
        plt.xticks(bins, np.round(bins, decimals=2))
        plt.xlabel('$\mathit{sim}_\mathit{cos}(p_i, g_i)$')
        plt.ylabel('count')
        plt.title('{}: similarity histogram'.format(algorithm))
        plt.savefig(sim_hist_plot_path.format(algorithm))
        logging.log(level=logging.INFO, msg='plotted similarity histogram of {}\n stored in {}'
                    .format(algorithm, sim_hist_plot_path.format(algorithm)))

        # plot cumulative similarity histogram
        plt.figure(1)
        plt.clf()
        axes = plt.gca()
        n, bins, _ = plt.hist(similarities[algorithm], bins=nbins, label=algorithm, color=algo2color(algorithm),
                              cumulative=True, normed=True)
        for idx, value in enumerate(n):
            plt.text(bins[idx], value+0.01, s=str(np.round(value, decimals=round_to)))
        axes.relim()
        axes.autoscale()
        plt.xticks(bins, np.round(bins, decimals=2))
        plt.xlabel('$\mathit{sim}_\mathit{cos}(p, g)$')
        plt.ylabel('$P[x \leq X]$')
        plt.title('{}: $P[x \leq X]$ similarity distribution'.format(algorithm))
        plt.savefig(leq_sim_distr_plot_path.format(algorithm))
        plt.clf()
        logging.log(level=logging.INFO, msg='plotted cumulative similarity distribution of {}\n stored in {}'
                    .format(algorithm, leq_sim_distr_plot_path.format(algorithm)))

    for combined_plot in combined_plots:
        # prepare similarities for combined histogram input
        sim = np.array(list(zip(*[similarities[algo] for algo in combined_plot])))

        # plot similarity histogram
        plt.figure(1)
        plt.clf()
        axes = plt.gca()
        n, bins, _ = plt.hist(sim, bins=nbins, label=combined_plot, color=[algo2color(algo) for algo in combined_plot])
        axes.relim()
        axes.autoscale()
        plt.xticks(bins, np.round(bins, decimals=2))
        plt.legend(loc=2)
        plt.xlabel('$\mathit{sim}_\mathit{cos}(p_i, g_i)$')
        plt.ylabel('count')
        plt.title('similarity histograms'.format(','.join(combined_plot)))
        plt.savefig(comb_sim_hist_plot_path.format('_'.join(combined_plot)))
        plt.clf()
        logging.log(level=logging.INFO, msg='plotted similarity histogram of {}\n stored in {}'
                    .format(','.join(combined_plot), comb_sim_hist_plot_path.format('_'.join(combined_plot))))

        # plot cumulative similarity distributions
        plt.figure(1)
        plt.clf()
        axes = plt.gca()
        n, bins, _ = plt.hist(sim, bins=nbins, label=combined_plot, color=[algo2color(algo) for algo in combined_plot],
                              cumulative=True, normed=True)
        axes.set_xlim([None, 1])
        axes.set_ylim([0, 1])
        plt.xticks(bins, np.round(bins, decimals=2))
        plt.legend(loc=2)
        plt.xlabel('$\mathit{sim}_\mathit{cos}(p, g)$')
        plt.ylabel('$P[x \leq X]$')
        plt.title('$P[x \leq X]$ similarity distribution'.format(','.join(combined_plot)))
        plt.savefig(comb_leq_sim_distr_plot_path.format('_'.join(combined_plot)))
        logging.log(level=logging.INFO, msg='plotted similarity distribution of {}\n stored in {}'
                    .format(','.join(combined_plot), comb_leq_sim_distr_plot_path.format('_'.join(combined_plot))))


if __name__ == '__main__':
    main()
