import json
import logging

from data_analysis.dumpio import JSONDumpReader
from data_analysis.utils import get_english_label


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        config = json.load(f)

    wikidata_path = config['wikidata dump']
    wikidata_labels_path = config['wikidata labels']

    with open(wikidata_labels_path, mode='w') as f:
        for e in JSONDumpReader(wikidata_path):
            if get_english_label(e):
                f.write(','.join([e['id'], get_english_label(e)]) + '\n')
    logging.log(level=logging.INFO, msg='wrote class ids to {}'.format(wikidata_labels_path))


if __name__ == '__main__':
    main()
