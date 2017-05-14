import json
import logging
import random

from algorithm.sequence_gen import DbGraphWalkSentences


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(process)d: %(message)s', level=logging.DEBUG)

    with open('algorithm_config.json') as f:
        gw_config = json.load(f)['graph walk sentences']

    with open('paths_config.json') as f:
        config = json.load(f)

    node_id_path = config['relevant class ids']
    edge_store_path = config['edges db']
    output_path = config['graph walk sentences']

    node_count = 10000
    offset = 2
    # computed: to 35000

    random.seed(1)

    with open(node_id_path) as f:
        nodes = [l.strip() for l in f]
    logging.log(level=logging.INFO, msg='node count: {}'.format(len(nodes)))
    logging.log(level=logging.INFO, msg='compute walks for {} nodes'.format(node_count))

    gen = DbGraphWalkSentences(
        nodes[30000:35000],
        depth=gw_config['depth'],  # RDF2Vec: depth = 4
        max_walks_per_v=gw_config['max walks'],  # RDF2Vec: max walks per vertice = 100
        edge_store_path=edge_store_path,
        workers=4
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
