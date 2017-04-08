import json
import logging

from data_analysis.class_extraction import analyze_characteristics
from data_analysis.dumpio import JSONDumpReader


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        config = json.load(f)

    orphan_classes_path = config['orphan class dump']
    characteristics_path = config['class characteristics']
    analysis_path = config['orphan class analysis']

    orphan_class_ids = set(map(lambda c: c['id'], JSONDumpReader(orphan_classes_path)))
    logging.log(level=logging.INFO, msg='loaded orphan class ids')

    with open(analysis_path, mode='w') as f:
        json.dump(
            analyze_characteristics(
                filter(lambda c: c['id'] in orphan_class_ids, JSONDumpReader(characteristics_path))
            ), f)
    logging.log(level=logging.INFO, msg='wrote analysis to {}'.format(analysis_path))

if __name__ == '__main__':
    main()
