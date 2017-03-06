from algorithm.sentence_gen import SentenceIterator


def main():
    node_id_path = '../data/algorithm_io/item_ids-20161107.txt'
    sentence_path = '../data/algorithm_io/graphwalk_sentences-20161107-1.txt'

    nodes = set()
    with open(node_id_path) as f:
        for l in f:
            nodes.add(l.strip())

    covered_nodes = set()
    for sentence in SentenceIterator(sentence_path):
        for word in sentence:
            if word[0] == 'Q':
                covered_nodes.add(word)

    print('coverage: {}/{} => {}%'.format(len(covered_nodes), len(nodes), 100*float(len(covered_nodes))/len(nodes)))


if __name__ == '__main__':
    main()
