# TODO: delete or refactor this.
import json
import itertools
from typing import List, Tuple

import data_analysis.config as config
import data_analysis.utils as utils


def sort_frequencies_as_tuples(frequencies: dict)->List[Tuple[str, int]]:
    return sorted(map(lambda t: (t[0], t[1]), frequencies.items()), key=lambda t: t[1], reverse=True)


def average(counts: dict)->float:
    a = 0
    s = 0
    for v, i in counts.items():
        a += i
        s += int(v) * i
    return float(s)/a


def main():
    chars1, chars2, chars3 = itertools.tee(utils.get_json_dicts(config.ROOT_CLASS_CHARACTERISTICS_PATH), 3)
    with open(config.ROOT_CLASS_ANALYSIS_PATH) as f:
        analysis = json.loads(f.readline())

    top_range = 20

    args = dict()
    args['root classes'] = analysis['root class count']

    args['labelless classes'] = len(list(filter(lambda r: r['label'] == '', chars1)))
    args['labeled classes'] = args['root classes'] - args['labelless classes']

    args['no wiki'] = len(list(filter(lambda r: r['enwiki'] == '' and r['simplewiki'] == '', chars2)))
    args['wiki'] = args['root classes'] - args['no wiki']

    args['average properties'] = average(analysis['property counts'])
    args['zero properties'] = analysis['property counts']['0']
    args['one property'] = analysis['property counts']['1']
    args['more properties'] = args['root classes'] - args['zero properties'] - args['one property']

    args['average instances'] = average(analysis['instance counts'])
    args['zero instances'] = analysis['instance counts']['0']
    args['one instance'] = analysis['instance counts']['1']
    args['more instances'] = args['root classes'] - args['zero instances'] - args['one instance']

    args['average subclasses'] = average(analysis['subclass counts'])
    args['zero subclasses'] = analysis['subclass counts']['0']
    args['one subclass'] = analysis['subclass counts']['1']
    args['more subclasses'] = args['root classes'] - args['zero subclasses'] - args['one subclass']

    property_frequencies = sort_frequencies_as_tuples(analysis['property frequencies'])
    for i in range(top_range):
        args['property {}'.format(i+1)], args['property {} occurrences'.format(i+1)] = property_frequencies[i]

    output = \
        '## summary of root class analysis\n' \
        '\n' \
        'root classes: {root classes}\n' \
        '\n' \
        '- without label: {labelless classes}\n' \
        '- with label: {labeled classes}\n' \
        '\n' \
        '- with wiki (en or simple) article: {wiki}\n' \
        '- without wiki (en or simple) article: {no wiki}\n' \
        '\n' \
        '- average properties per class: {average properties}\n' \
        '- with 0 properties: {zero properties}\n' \
        '- with 1 property: {one property}\n' \
        '- with more than 1 property: {more properties}\n' \
        '\n' \
        '- average instances per class: {average instances}\n' \
        '- with 0 instances: {zero instances}\n' \
        '- with 1 instance: {one instance}\n' \
        '- with more than 1 instance: {more instances}\n' \
        '\n' \
        '- average subclasses per class: {average subclasses}\n' \
        '- with 0 subclasses: {zero subclasses}\n' \
        '- with 1 subclass: {one subclass}\n' \
        '- more than 1 subclass: {more subclasses}\n' \
        '\n' \
        'most frequent properties\n'\
        .format(**args)

    output += ''.join(['{}. {} with {}\n'
                      .format(i, args['property {}'.format(i)],
                              args['property {} occurrences'.format(i)]) for i in range(1, top_range+1)]) + '\n'

    output += '\n list of all root classes: \n'
    for ch in chars3:
        output += '{}({}):enwiki: {}, simplewiki: {}, properties: {}; subclasses: {}; instances: {}\n'.format(
            ch.get('label'), ch.get('id'), ch.get('enwiki'), ch.get('simplewiki'), ' ,'.join(ch.get('properties')),
            ', '.join(ch.get('subclasses')), ', '.join(ch.get('instances'))
        )

    with open('output/summary.txt', 'w') as f:
        f.write(output)


if __name__ == '__main__':
    main()
