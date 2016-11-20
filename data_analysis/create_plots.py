# TODO: make the figures better
# TODO: save the figures with plt.savefig('.svg')
import json
from typing import List, Tuple

import matplotlib.pyplot as plt

import data_analysis.config as config


def sort_counts_as_tuples(counts: dict)->List[Tuple[int, int]]:
    return sorted(map(lambda t: (int(t[0]), t[1]), counts.items()), key=lambda t: t[0])


def sort_frequencies_as_tuples(frequencies: dict)->List[Tuple[str, int]]:
    return sorted(map(lambda t: (t[0], t[1]), frequencies.items()), key=lambda t: t[1], reverse=True)


def main():
    with open(config.ROOT_CLASS_ANALYSIS_PATH) as f:
        analysis = json.loads(f.readline())

    root_class_count = analysis.get('root class count')
    property_counts = sort_counts_as_tuples(analysis.get('property counts'))
    subclass_counts = sort_counts_as_tuples(analysis.get('subclass counts'))
    instance_counts = sort_counts_as_tuples(analysis.get('instance counts'))
    property_frequencies = sort_frequencies_as_tuples(analysis.get('property frequencies'))
    topic_frequencies = sort_frequencies_as_tuples(analysis.get('topic frequencies'))

    print('number of root classes: {}'.format(root_class_count))

    plt.figure(1)
    plt.bar([i[0] for i in property_counts], [i[1] for i in property_counts])
    plt.xlabel('sum of properties')
    plt.ylabel('number of classes')

    plt.figure(2)
    plt.bar([i[0] for i in subclass_counts], [i[1] for i in subclass_counts])
    plt.xlabel('sum of subclasses')
    plt.ylabel('number of classes')
    plt.show()

    plt.figure(3)
    plt.bar([i[0] for i in instance_counts], [i[1] for i in instance_counts])
    plt.xlabel('sum of instances')
    plt.ylabel('number of classes')
    plt.show()

    plt.figure(3)
    plt.bar([i[0] for i in instance_counts], [i[1] for i in instance_counts])
    plt.xlabel('sum of instances')
    plt.ylabel('number of classes')
    plt.show()


if __name__ == '__main__':
    main()


