import logging

import data_analysis.dumpio as dumpio
import data_analysis.pipe as pipe
from data_analysis.class_extraction import get_class_ids, is_item, is_unlinked_class

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main():
    # Reads from Wikidata JSON dump.
    entity_dump_path = '../data/wikidata/wikidata-20161107-all.json'
    # Writes and then reads JSON dump containing only classes.
    class_dump_path = '../data/classes-20161107'
    # Writes JSON dump containing only unlinked classes.
    unlinked_class_dump_path = '../data/unlinked_classes-20161107'

    # get the set of ids, which identify classes in dump
    class_ids = get_class_ids(dumpio.JSONDumpReader(entity_dump_path))
    is_class = lambda e: e['id'] in class_ids
    logging.log(level=logging.INFO, msg='found {} classes'.format(len(class_ids)))

    # write all classes into new JSON dump
    pipe.DataPipe.read_json_dump(entity_dump_path).filter_by(is_item).filter_by(is_class).write(class_dump_path)
    logging.log(level=logging.INFO, msg='finished writing classes')

    # write all unlinked classes into new JSON dump
    pipe.DataPipe.read_json_dump(class_dump_path).filter_by(is_unlinked_class).write(unlinked_class_dump_path)
    logging.log(level=logging.INFO, msg='finished writing unlinked classes')

if __name__ == '__main__':
    main()
