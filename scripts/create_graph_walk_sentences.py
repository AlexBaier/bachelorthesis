import logging

from algorithm.sentence_gen import GraphWalkSentences

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main():
    node_id_path = '../data/algorithm_io/class_ids-20161107.txt'
    edge_store_path = '../data/algorithm_io/edges.sqlite3'
    output_path = '../data/algorithm_io/graphwalk_sentences-20161107.txt'
    nodes = list()
    with open(node_id_path) as f:
        for l in f:
            nodes.append(l.strip())
    logging.log(level=logging.INFO, msg='node count: {}'.format(len(nodes)))
    gen = GraphWalkSentences(
        nodes,
        8,  # RDF2Vec uses 8, 4 is used for runtime
        edge_store_path
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
