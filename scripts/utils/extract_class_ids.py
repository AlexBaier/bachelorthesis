import json
import logging

from data_analysis.dumpio import JSONDumpReader


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        config = json.load(f)

    classes_path = config['class dump']
    class_ids_path = config['class ids']

    with open(class_ids_path, mode='w') as f:
        f.write('\n'.join(map(lambda c: c['id'], JSONDumpReader(classes_path))) + '\n')
    logging.log(level=logging.INFO, msg='wrote class ids to {}'.format(class_ids_path))


if __name__ == '__main__':
    main()
