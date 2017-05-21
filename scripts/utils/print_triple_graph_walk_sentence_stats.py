import json
import logging

from data_analysis.dumpio import JSONDumpReader


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        config = json.load(f)

    triple_sentences_path = config['triple sentences']
    graph_walk_sentences_path = config['graph walk sentences']
    relevant_class_ids_path = config['relevant class ids']
    wikidata_dump_path = config['wikidata dump']

    n_triple_sentences = 0

    with open(triple_sentences_path) as f:
        for _ in f:
            n_triple_sentences += 1
    logging.info('loaded triple sentences')

    n_graph_walk_sentences = 0
    unique_sources = set()
    unique_items = set()

    with open(graph_walk_sentences_path) as f:
        for sentence in map(lambda l: l.strip().split(), f):
            n_graph_walk_sentences += 1
            unique_sources.add(sentence[0])
            unique_items.update(sentence[::2])
    logging.info('loaded graph walk sentences')

    n_unique_sources = len(unique_sources)
    n_unique_items = len(unique_items)

    relevant_classes = set()

    with open(relevant_class_ids_path) as f:
        for cid in map(lambda l: l.strip(), f):
            relevant_classes.add(cid)
    logging.info('loaded relevant class ids')

    n_relevant_classes = len(relevant_classes)

    all_items = set()

    for entity in JSONDumpReader(wikidata_dump_path):
        if entity['id'][0] == 'Q':
            all_items.add(entity['id'])
    logging.info('loaded all items')

    n_all_items = len(all_items)

    relevant_class_coverage = len(relevant_classes.intersection(unique_items)) / float(n_relevant_classes)
    all_item_coverage = float(n_unique_items) / n_all_items

    print('triple_sentences = {}'.format(n_triple_sentences))
    print()
    print('graph walk sentences:')
    print('graph walk sentences = {}'.format(n_graph_walk_sentences))
    print('unique sources = {}'.format(n_unique_sources))
    print('unique_items = {}'.format(n_unique_items))
    print('relevant class coverage = {:.2f}%'.format(100.0*relevant_class_coverage))
    print('all item coverage = {:2f}%'.format(100.0*all_item_coverage))


if __name__ == '__main__':
    main()
