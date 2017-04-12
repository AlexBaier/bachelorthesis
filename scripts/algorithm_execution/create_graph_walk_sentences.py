import json
import logging
import random
import sys

from typing import List, Tuple

from algorithm.sequence_gen import GraphWalkSentences


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(process)d: %(message)s', level=logging.DEBUG)

    with open('algorithm_config.json') as f:
        gw_config = json.load(f)['graph walk sentences']

    with open('paths_config.json') as f:
        config = json.load(f)

    node_id_path = config['relevant class ids']
    triple_sentence_path = config['triple sentences']
    output_path = config['graph walk sentences']

    min_source = -1
    max_source = -1
    with open(triple_sentence_path) as f:
        for ft in map(lambda t: list(map(lambda k: int(k[1:]), t)),
                      filter(lambda s: len(s) == 3, map(lambda l: l.strip().split(), f))):
            if ft[0] > max_source:
                max_source = ft[0]
            if ft[0] < min_source or min_source == -1:
                min_source = ft[0]

    edges = [list() for _ in range(max_source - min_source+1)]  # type: List[List[Tuple[int, int]]]
    with open(triple_sentence_path) as f:
        for ft in map(lambda t: list(map(lambda k: int(k[1:]), t)),
                      filter(lambda s: len(s) == 3, map(lambda l: l.strip().split(), f))):
            edges[ft[0]-min_source].append((ft[1], ft[2]))
    logging.log(level=logging.INFO, msg='loaded edges, memory usage={} MByte'.format(sys.getsizeof(edges)*10e-6))

    def get_out_edges(node_id: str)->List[Tuple[str, str]]:
        node_id = int(node_id[1:])
        return list(map(lambda out: ('P' + str(out[0]), 'Q' + str(out[1])), edges[node_id-min_source]))

    node_count = 10

    random.seed()

    with open(node_id_path) as f:
        nodes = [l.strip() for l in f]
    logging.log(level=logging.INFO, msg='node count: {}'.format(len(nodes)))

    random.shuffle(nodes)
    gen = GraphWalkSentences(
        nodes[:node_count],
        depth=gw_config['depth'],  # RDF2Vec: depth = 4
        max_walks_per_v=gw_config['max walks'],  # RDF2Vec: max walks per vertice = 100
        get_out_edges=get_out_edges
    )
    logging.log(level=logging.INFO, msg='initialized graph walk sentence gen')

    sentences = gen.get_sequences()
    with open(output_path, mode='a') as f:
        c = 1
        for sentence in map(lambda s: ' '.join(s) + '\n', sentences):
            f.write(sentence)
            if c % 1000 == 0:
                print('appended', str(c), 'sentences')
            c += 1

if __name__ == '__main__':
    main()
