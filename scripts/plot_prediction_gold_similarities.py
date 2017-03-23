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

    sim_sample_plot_path = '../evaluation/plots/sim_sample_{}-20161107.png'
    leq_sim_distr_plot_path = '../evaluation/plots/le_sim_distr_{}-20161107.png'
    geq_sim_distr_plot_path = '../evaluation/plots/ge_sim_distr_{}-20161107.png'

    comb_sim_sample_plot_path = '../evaluation/plots/comb_sim_sample_{}-20161107.png'
    comb_leq_sim_distr_plot_path = '../evaluation/plots/comb_le_sim_distr_{}-20161107.png'
    comb_geq_sim_distr_plot_path = '../evaluation/plots/comb_ge_sim_distr_{}-20161107.png'

    algorithms = [
        'baseline',
        'distknn',
        'linproj',
        'pwlinproj'
    ]
    combined_plots = [
        ['baseline', 'pwlinproj'],
        ['linproj', 'pwlinproj'],
        ['baseline', 'linproj', 'pwlinproj']
    ]

    steps = 11
    round_to = 3

    with open(config_path) as f:
        config = json.load(f)
    logging.log(level=logging.INFO, msg='loaded algorithm config')

    golds = load_test_data(test_data_path)
    gold_count = len(golds)
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
        plt.figure(1)
        axes = plt.gca()
        axes.set_ylim([0, 1])
        # plot density function
        axes.set_xlim([0, gold_count])
        x_count = np.arange(1, gold_count+1)
        y_sim = np.sort(similarities[algorithm])
        plt.plot(x_count, y_sim, linewidth=6, label=algorithm, color=algo2color(algorithm))
        plt.xlabel('$i$')
        plt.ylabel('$\mathit{sim}_\mathit{cos}(p_i, g_i)$')
        plt.title('{}: similarity for all prediction-gold pairs'.format(algorithm))
        plt.savefig(sim_sample_plot_path.format(algorithm))
        plt.clf()
        logging.log(level=logging.INFO, msg='plotted similarity samples of {}\n stored in {}'
                    .format(algorithm, sim_sample_plot_path.format(algorithm)))
        # plot similarity distributions
        axes.set_xlim([0, 1])
        x_sim = np.linspace(0, 1, num=steps)
        y_prob = np.zeros(steps)
        # plot leq similarity distribution
        for idx, sim in enumerate(x_sim):
            y_prob[idx] = np.count_nonzero(similarities[algorithm] <= sim) / gold_count
        plt.bar(np.arange(0, steps), y_prob, alpha=1, label=algorithm, color=algo2color(algorithm))
        plt.xticks(np.arange(0, steps) + 0.5, np.round(x_sim, decimals=round_to))
        plt.xlabel('$\mathit{sim}_\mathit{cos}(p, g)$')
        plt.ylabel('$P[x \leq X]$')
        plt.title('{}: $P[x \leq X]$ similarity distribution'.format(algorithm))
        plt.savefig(leq_sim_distr_plot_path.format(algorithm))
        plt.clf()
        logging.log(level=logging.INFO, msg='plotted similarity distribution of {}\n stored in {}'
                    .format(algorithm, leq_sim_distr_plot_path.format(algorithm)))

        # plot geq similarity distribution
        for idx, sim in enumerate(x_sim):
            y_prob[idx] = np.count_nonzero(similarities[algorithm] >= sim) / gold_count
        plt.bar(np.arange(0, steps), y_prob, alpha=1, label=algorithm, color=algo2color(algorithm))
        plt.xticks(np.arange(0, steps) + 0.5, np.round(x_sim, decimals=round_to))
        plt.xlabel('$\mathit{sim}_\mathit{cos}(p, g)$')
        plt.ylabel('$P[x \geq X]$')
        plt.title('{}: $P[x \geq X]$ similarity distribution'.format(algorithm))
        plt.savefig(geq_sim_distr_plot_path.format(algorithm))
        plt.clf()
        logging.log(level=logging.INFO, msg='plotted similarity distribution of {}\n stored in {}'
                    .format(algorithm, geq_sim_distr_plot_path.format(algorithm)))

    for combined_plot in combined_plots:
        width = 1.0/(len(combined_plot)+1)
        step_offset = 1.0/len(combined_plot)

        plt.figure(1)
        axes = plt.gca()
        axes.set_ylim([0, 1])

        # plot density function
        axes.set_xlim([0, gold_count])
        legend_handles = list()
        x_count = np.arange(1, gold_count + 1)
        for algorithm in combined_plot:
            y_sim = np.sort(similarities[algorithm])
            rect = plt.plot(x_count, y_sim, linewidth=6, label=algorithm, color=algo2color(algorithm))
            legend_handles.append(rect[0])
        plt.xlabel('$i$')
        plt.ylabel('$\mathit{sim}_\mathit{cos}(p_i, g_i)$')
        plt.title('similarity for all prediction-gold pairs'.format(','.join(combined_plot)))
        plt.legend(legend_handles, combined_plot)
        plt.savefig(comb_sim_sample_plot_path.format('_'.join(combined_plot)))
        plt.clf()
        logging.log(level=logging.INFO, msg='plotted similarity samples of {}\n stored in {}'
                    .format(','.join(combined_plot), comb_sim_sample_plot_path.format('_'.join(combined_plot))))

        # plot similarity distributions
        axes.set_xlim([0, 1])
        x_sim = np.linspace(0, 1, num=steps)
        # plot leq similarity distribution
        legend_handles = list()
        for bar_idx, algorithm in enumerate(combined_plot):
            y_prob = np.zeros(steps)
            for idx, sim in enumerate(x_sim):
                y_prob[idx] = np.count_nonzero(similarities[algorithm] <= sim) / gold_count
            rect = plt.bar(np.arange(0, steps) + bar_idx*(step_offset if bar_idx == 0 else width),
                           y_prob, width=width, linewidth=1, fill=True, label=algorithm, color=algo2color(algorithm),
                           edgecolor=algo2color(algorithm))
            legend_handles.append(rect[0])
        plt.xticks(np.arange(0, steps) + 0.35, np.round(x_sim, decimals=round_to))
        plt.xlabel('$\mathit{sim}_\mathit{cos}(p, g)$')
        plt.ylabel('$P[x \leq X]$')
        plt.title('$P[x \leq X]$ similarity distribution'.format(','.join(combined_plot)))
        plt.legend(legend_handles, combined_plot)
        plt.savefig(comb_leq_sim_distr_plot_path.format('_'.join(combined_plot)))
        plt.clf()
        logging.log(level=logging.INFO, msg='plotted similarity distribution of {}\n stored in {}'
                    .format(','.join(combined_plot), comb_leq_sim_distr_plot_path.format('_'.join(combined_plot))))

        # plot geq similarity distribution
        legend_handles = list()
        for bar_idx, algorithm in enumerate(combined_plot):
            y_prob = np.zeros(steps)
            for idx, sim in enumerate(x_sim):
                y_prob[idx] = np.count_nonzero(similarities[algorithm] >= sim) / gold_count
            rect = plt.bar(np.arange(0, steps) + bar_idx*(step_offset if bar_idx == 0 else width),
                           y_prob, width=width, linewidth=1, fill=True, label=algorithm, color=algo2color(algorithm),
                           edgecolor=algo2color(algorithm))
            legend_handles.append(rect[0])
        plt.xticks(np.arange(0, steps) + 0.35, np.round(x_sim, decimals=round_to))
        plt.xlabel('$\mathit{sim}_\mathit{cos}(p, g)$')
        plt.ylabel('$P[x \geq X]$')
        plt.title('$P[x \geq X]$ similarity distribution'.format(','.join(combined_plot)))
        plt.legend(legend_handles, combined_plot)
        plt.savefig(comb_geq_sim_distr_plot_path.format('_'.join(combined_plot)))
        plt.clf()
        logging.log(level=logging.INFO, msg='plotted similarity distribution of {}\n stored in {}'
                    .format(','.join(combined_plot), comb_geq_sim_distr_plot_path.format('_'.join(combined_plot))))

if __name__ == '__main__':
    main()
