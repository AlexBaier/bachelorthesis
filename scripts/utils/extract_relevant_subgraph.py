import json
import logging
import random

from data_analysis import dumpio


def collect_nodes(subjects, triples_path):
    nodes = set()
    with open(triples_path) as f:
        for subj, _, obj in filter(lambda t: len(t) == 3, map(lambda l: l.strip().split(), f)):
            if subj in subjects:
                nodes.add(obj)
    return nodes


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    depth = 4
    assert depth >= 2

    with open('paths_config.json') as f:
        config = json.load(f)

    class_ids_path = config['relevant class ids']
    characteristics_path = config['class characteristics']
    triples_path = config['triple sentences']
    subgraph_path = config['subgraph triples']

    with open(class_ids_path) as f:
        class_ids = set(l.strip() for l in f.readlines())

    # subgraph contains all classes
    nodes = class_ids.copy()
    # subgraph contains instances of classes
    random.seed(1)
    for characteristic in dumpio.JSONDumpReader(characteristics_path):
        if characteristic['id'] in class_ids:
            nodes.update(characteristic['instances'])

    current_depth = 0

    while current_depth < depth - 1:
        logging.info('{} nodes at depth {}'.format(len(nodes), current_depth))
        nodes.update(collect_nodes(nodes, triples_path))
        current_depth += 1
    logging.info('{} nodes at depth {}'.format(len(nodes), current_depth))

    edge_count = 0
    all_nodes = nodes.copy()
    with open(triples_path) as f, open(subgraph_path, mode='w') as g:
        for subj, pred, obj in filter(lambda t: len(t) == 3 and t[0] in nodes, map(lambda l: l.strip().split(), f)):
            g.write(' '.join((subj, pred, obj)) + '\n')
            edge_count += 1
            all_nodes.add(obj)
    logging.info('subgraph contains {} nodes and {} edges'.format(len(all_nodes), edge_count))
    logging.info('wrote subgraph edges to {}'.format(subgraph_path))

if __name__ == '__main__':
    main()
