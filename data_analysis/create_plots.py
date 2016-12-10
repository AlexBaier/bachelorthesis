import itertools
import json
from typing import List, Tuple

import matplotlib.pyplot as plt

import data_analysis.config as config


def sort_counts_as_tuples(counts: dict)->List[Tuple[int, int]]:
    return sorted(map(lambda t: (int(t[0]), t[1]), counts.items()), key=lambda t: t[0])


def sort_frequencies_as_tuples(frequencies: dict)->List[Tuple[str, int]]:
    return sorted(map(lambda t: (t[0], t[1]), frequencies.items()), key=lambda t: t[1], reverse=True)


def create_plots(analysis_path: str, property_sum_fig: str, instance_sum_fig: str, subclass_sum_fig: str,
                 property_frequency_fig: str):
    with open(analysis_path) as f:
        analysis = json.loads(f.readline())

    root_class_count = analysis.get('root class count')
    property_counts = sort_counts_as_tuples(analysis.get('property counts'))
    subclass_counts = sort_counts_as_tuples(analysis.get('subclass counts'))
    instance_counts = sort_counts_as_tuples(analysis.get('instance counts'))
    property_frequencies = sort_frequencies_as_tuples(analysis.get('property frequencies'))
    # topic_frequencies = sort_frequencies_as_tuples(analysis.get('topic frequencies'))

    print('number of root classes: {}'.format(root_class_count))

    # property sum/number of classes figure
    plt.figure(1)
    x = [i[0] for i in property_counts]
    y = [i[1] for i in property_counts]
    plt.bar(x, y)
    # set x axis steps
    plt.xlim(0, 20)
    plt.title('number of classes with specific amount of properties')
    plt.xlabel('sum of properties')
    plt.ylabel('number of classes')
    # set values on bars
    for i, v in itertools.islice(enumerate(y), 20):
        plt.text(i, v, str(v), color='blue', fontweight='bold')

    plt.savefig(property_sum_fig)

    # subclass sum/number of classes figure
    plt.figure(2)
    x = [i[0] for i in subclass_counts]
    y = [i[1] for i in subclass_counts]
    plt.xlim(0, 20)
    plt.bar(x, y)
    plt.title('number of classes with a specific amount of subclasses')
    plt.xlabel('sum of subclasses')
    plt.ylabel('number of classes')
    # set values on bars
    for i, v in itertools.islice(enumerate(y), 20):
        plt.text(i, v, str(v), color='blue', fontweight='bold')

    plt.savefig(subclass_sum_fig)

    # instance sum/number of classes figure
    plt.figure(3)
    x = [i[0] for i in instance_counts]
    y = [i[1] for i in instance_counts]
    plt.xlim(0, 20)
    #plt.yticks(y[:6])
    plt.bar(x, y)
    plt.title('number of classes with specific amount of instances')
    plt.xlabel('sum of instances')
    plt.ylabel('number of classes')
    for i, v in itertools.islice(enumerate(y), 20):
        plt.text(i, v, str(v), color='blue', fontweight='bold')

    plt.savefig(instance_sum_fig)

    # property/frequency figure
    plt.figure(4)
    limiter = 14
    x_values = [i[0] for i in property_frequencies][:limiter]
    x = list(range(0, len(x_values)))
    y = [i[1] for i in property_frequencies][:limiter]
    plt.xlim(0, limiter)
    plt.xticks(x, x_values)
    plt.bar(x, y)
    plt.title('frequency of properties')
    plt.xlabel('property')
    plt.ylabel('occurrences')
    for i, v in enumerate(y):
        plt.text(i, v, str(v), color='blue', fontweight='bold')

    plt.savefig(property_frequency_fig)


def main():
    create_plots(config.ROOT_CLASS_ANALYSIS_PATH, config.PROPERTY_SUM_FIGURE_PATH, config.INSTANCE_SUM_FIGURE_PATH,
                 config.SUBCLASS_SUM_FIGURE_PATH, config.PROPERTY_FREQUENCY_FIGURE_PATH)

if __name__ == '__main__':
    main()
