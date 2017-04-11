import json
import logging

from data_analysis.dumpio import JSONDumpReader
from data_analysis.utils import get_subclass_of_ids


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        config = json.load(f)

    classes_path = config['class dump']
    subclass_of_path = config['subclass of relations']

    with open(subclass_of_path, mode='w') as f:
        f.writelines(','.join([c['id']] + list(get_subclass_of_ids(c))) + '\n' for c in JSONDumpReader(classes_path))
    logging.log(level=logging.INFO, msg='wrote subclass of relations to {}'.format(subclass_of_path))


if __name__ == '__main__':
    main()
