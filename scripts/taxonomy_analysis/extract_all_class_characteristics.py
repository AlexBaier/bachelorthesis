import json
import logging

from data_analysis.class_extraction import to_characteristic
from data_analysis.dumpio import JSONDumpReader, JSONDumpWriter


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        config = json.load(f)

    wikidata_path = config['wikidata dump']
    class_ids_path = config['class ids']
    classes_path = config['classes dump']
    characteristics_path = config['class characteristics']

    with open(class_ids_path) as f:
        class_ids = set(map(lambda l: l.strip(), f.readlines()))
    logging.log(level=logging.INFO, msg='loaded class ids')

    to_charac = to_characteristic(class_ids, JSONDumpReader(wikidata_path))
    logging.log(level=logging.INFO, msg='computed subclasses and instances of all classes')

    JSONDumpWriter(characteristics_path).write(
        map(lambda ch: ch.to_dict(), map(to_charac, JSONDumpReader(classes_path)))
    )
    logging.log(level=logging.INFO, msg='wrote characteristics to {}'.format(characteristics_path))

if __name__ == '__main__':
    main()
