import json
import logging

from data_analysis.dumpio import JSONDumpReader, JSONDumpWriter


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    with open(paths_config['irrelevant properties']) as f:
        irrelevant_properties = set(l.strip() for l in f)

    classes_path = paths_config['class dump']
    relevant_classes_path = paths_config['relevant class dump']

    JSONDumpWriter(relevant_classes_path).write(
        filter(lambda c: set(c['claims'].keys()).isdisjoint(irrelevant_properties), JSONDumpReader(classes_path)))
    logging.log(level=logging.INFO, msg='wrote relevant classes to {}'.format(relevant_classes_path))

if __name__ == '__main__':
    main()
