import json
import logging

from data_analysis.class_extraction import get_class_ids, is_item, is_orphan_class
from data_analysis.dumpio import JSONDumpReader, JSONDumpWriter


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        config = json.load(f)

    # Reads from Wikidata JSON dump.
    wikidata_path = config['wikidata dump']
    # Writes and then reads JSON dump containing only classes.
    class_dump_path = config['class dump']
    # Writes JSON dump containing only orphan classes.
    orphan_class_dump_path = config['orphan class dump']

    # get the set of ids, which identify classes in dump
    class_ids = get_class_ids(JSONDumpReader(wikidata_path))
    logging.log(level=logging.INFO, msg='found {} classes'.format(len(class_ids)))

    # write all classes into new JSON dump
    JSONDumpWriter(class_dump_path).write(
        filter(lambda e: is_item(e) and e['id'] in class_ids, JSONDumpReader(wikidata_path))
    )
    logging.log(level=logging.INFO, msg='wrote classes to {}'.format(class_dump_path))

    # write all unlinked classes into new JSON dump
    JSONDumpWriter(orphan_class_dump_path).write(
        filter(is_orphan_class, JSONDumpReader(class_dump_path))
    )
    logging.log(level=logging.INFO, msg='wrote orphan classes to {}'.format(orphan_class_dump_path))

if __name__ == '__main__':
    main()
