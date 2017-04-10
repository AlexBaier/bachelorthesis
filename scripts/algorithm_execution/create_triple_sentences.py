import json
import logging

from typing import List

from algorithm.sequence_gen import TripleSentences
from data_analysis.dumpio import JSONDumpReader
from evaluation.utils import load_test_inputs


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        config = json.load(f)

    dump_path = config['wikidata dump']
    class_charac_path = config['class characteristics']
    test_data_path = config['test data']
    output_path = config['triple sentences']

    with open(config['irrelevant properties']) as f:
        irrelevant_properties = set(l.strip() for l in f)

    with open(config['relevant class ids']) as f:
        relevant_class_ids = set(l.strip() for l in f)

    relevant_item_ids = set()
    for charac in filter(lambda c: c['id'] in relevant_class_ids, JSONDumpReader(class_charac_path)):
        relevant_item_ids.update(charac['instances'])
        relevant_item_ids.update((charac['subclasses']))
        relevant_item_ids.add(charac['id'])
    logging.log(level=logging.INFO, msg='identified {} relevant itemds'.format(len(relevant_item_ids)))

    # triple sentences should not include the test samples
    sources = set(load_test_inputs(test_data_path))
    relation = 'P279'

    def is_forbidden_triple(triple: List[str])->bool:
        if triple[1] == relation and triple[0] in sources:
            return True
        return False

    sentences = TripleSentences(
        filter(lambda o: o['id'] in relevant_item_ids, JSONDumpReader(dump_path)),
        forbidden_properties=irrelevant_properties,
        is_forbidden_triple=is_forbidden_triple
    ).get_sequences()

    with open(output_path, mode='w') as f:
        for idx, sentence in enumerate(map(lambda s: ' '.join(s) + '\n', sentences)):
            f.write(sentence)
            if idx % 10000 == 0:
                logging.log(level=logging.INFO, msg='wrote {} sentences'.format(idx + 1))
    logging.log(level=logging.INFO, msg='wrote triple sentences to {}'.format(output_path))

if __name__ == '__main__':
    main()
