import json
import logging
import random

from algorithm.sequence_gen import GraphWalkSentences


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('algorithm_config.json') as f:
        gw_config = json.load(f)['graph walk sentences']

    with open('paths_config.json') as f:
        config = json.load(f)

    subgraph_path = config['subgraph triples']
    output_path = config['graph walk sentences']

    n_sources = 5000000

    source_ids = set()
    with open(subgraph_path) as f:
        for subj, _, _ in map(lambda l: l.strip().split(), f):
            source_ids.add(subj)
    source_ids = set(random.sample(source_ids, n_sources))

    subgraph_nodes = source_ids.copy()
    for current_depth in range(1, gw_config['depth'] + 1):
        with open(subgraph_path) as f:
            temp_nodes = set()
            for subj, _, obj in map(lambda l: l.strip().split(), f):
                if subj in subgraph_nodes:
                    temp_nodes.add(obj)
            subgraph_nodes.update(temp_nodes)
    logging.info('{} nodes in subgraph'.format(len(subgraph_nodes)))

    id2idx = dict()
    current_idx = 0
    edges = list()
    c = 0
    with open(subgraph_path) as f:
        for subj, pred, obj in map(lambda l: l.strip().split(), f):
            if subj not in subgraph_nodes:
                continue
            if not id2idx.get(subj, None):
                id2idx[subj] = current_idx
                edges.append(set())
                current_idx += 1
            edges[id2idx[subj]].add((pred, obj))
            c += 1
            if c % 500000 == 0:
                logging.info('{} edges'.format(c))

    def get_out_edges(node_id):
        if id2idx.get(node_id, None):
            return list(edges[id2idx[node_id]])
        else:
            return list()

    # write triples
    with open(subgraph_path) as f, open(output_path, mode='w') as g:
        for subj, pred, obj in map(lambda l: l.strip().split(), f):
            g.write('{} {} {}\n'.format(subj, pred, obj))

    # write graph walks
    chunks = 50000
    progress = 0
    for node_ids in [list(source_ids)[i:i+chunks] for i in range(0, len(source_ids), chunks)]:
        gen = GraphWalkSentences(
            node_ids,
            depth=gw_config['depth'],  # RDF2Vec: depth = 4
            max_walks_per_v=gw_config['max walks'],  # RDF2Vec: max walks per vertice = 100
            get_out_edges=get_out_edges
        )

        sentences = gen.get_sequences()
        with open(output_path, mode='a') as f:
            c = 0
            for sentence in map(lambda s: ' '.join(s) + '\n', sentences):
                f.write(sentence)
                c += 1
        progress += 1
        logging.info('chunks: {}/{}'.format(progress, int(len(source_ids)/chunks)+1))
    logging.info('wrote graph walks to {}'.format(output_path))

if __name__ == '__main__':
    main()
