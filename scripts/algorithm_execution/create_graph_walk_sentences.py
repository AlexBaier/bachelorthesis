import json
import logging

from algorithm.sequence_gen import GraphWalkSentences


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('algorithm_config.json') as f:
        gw_config = json.load(f)['graph walk sentences']

    with open('paths_config.json') as f:
        config = json.load(f)

    subgraph_path = config['subgraph triples']
    output_path = config['graph walk sentences']

    source_ids = set()
    id2idx = dict()
    current_idx = 0
    edges = list()
    c = 0
    with open(subgraph_path) as f:
        for subj, pred, obj in filter(lambda t: len(t) == 3, map(lambda l: l.strip().split(), f)):
            if not id2idx.get(subj, None):
                id2idx[subj] = current_idx
                edges.append(list())
                current_idx += 1
            edges[id2idx[subj]].append((pred, obj))
            source_ids.add(subj)
            c += 1
            if c % 500000 == 0:
                logging.info('{} edges'.format(c))
    source_ids = list(source_ids)

    def get_out_edges(node_id):
        if id2idx.get(node_id, None):
            return edges[id2idx[node_id]]
        else:
            return list()

    chunks = 5000
    progress = 0
    for node_ids in [source_ids[i:i+chunks] for i in range(0, len(source_ids), chunks)]:
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
