from algorithm.sentence_gen import SentenceIterator


def main():
    node_id_path = '../data/algorithm_io/class_ids-20161107.txt'
    sentence_path = '../data/algorithm_io/graphwalk_sentences-20161107.txt'

    nodes = set()
    with open(node_id_path) as f:
        for l in f:
            nodes.add(l.strip())

    with open(sentence_path) as f:
        covered_nodes = set(filter(lambda w: w[0] == 'Q', f.read().replace('\n', ' ').split()))

    print('coverage: {}/{} => {}%'.format(len(covered_nodes), len(nodes), 100*float(len(covered_nodes))/len(nodes)))


if __name__ == '__main__':
    main()
