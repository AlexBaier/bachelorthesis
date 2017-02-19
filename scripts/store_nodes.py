import logging
from data_analysis.dumpio import JSONDumpReader


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main():
    dump_path = '../data/wikidata/wikidata-20161107-all.json'
    all_nodes_path = '../data/algorithm_io/item_ids-20161107.txt'
    nodes = list()
    for o in JSONDumpReader(dump_path):
        if o['id'][0] == 'Q':
            nodes.append(o['id'])
    print('collected {} nodes'.format(len(nodes)))
    with open(all_nodes_path, mode='w') as f:
        for n in nodes:
            f.write(n + '\n')
    print('stored nodes')

if __name__ == '__main__':
    main()
