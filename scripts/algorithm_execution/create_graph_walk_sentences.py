import json
import logging
import random

from algorithm.sequence_gen import GraphWalkSentences


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(process)d: %(message)s', level=logging.DEBUG)

    with open('paths_config.json') as f:
        config = json.load(f)

    node_id_path = config['class ids']
    edge_store_path = config['edges db']
    output_path = config['graph walk sentences']

    node_count = 2500

    random.seed()

    with open(node_id_path) as f:
        nodes = [l.strip() for l in f]
    logging.log(level=logging.INFO, msg='node count: {}'.format(len(nodes)))

    random.shuffle(nodes)
    gen = GraphWalkSentences(
        nodes[:node_count],
        depth=4,  # RDF2Vec: depth = 4
        max_walks_per_v=10,  # RDF2Vec: max walks per vertice = 100
        edge_store_path=edge_store_path,
        workers=4
    )
    logging.log(level=logging.INFO, msg='initialized graph walk sentence gen')

    sentences = gen.get_sequences()
    with open(output_path, mode='w') as f:
        c = 1
        for sentence in map(lambda s: ' '.join(s) + '\n', sentences):
            f.write(sentence)
            if c % 1000 == 0:
                print('written', str(c), 'sentences')
            c += 1

if __name__ == '__main__':
    main()
