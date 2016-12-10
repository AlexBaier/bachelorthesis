"""
this module uses the output of reduce_json_dump.
"""
import json
from typing import Iterable, Set

import data_analysis.config as config
from data_analysis.constants import INSTANCE_OF, SUBCLASS_OF, ID
import data_analysis.utils as utils


def get_class_ids(reduced_items: Iterable[dict])->Set[str]:
    """
    :param reduced_items:
    :return:
    """
    classes = set()  # type: Set[str]
    for idx, item in enumerate(reduced_items):
        if item.get(SUBCLASS_OF):  # if an item has subclass of and instance of, it is still considered a class.
            classes.add(item.get(ID))
            classes.update(item.get(SUBCLASS_OF))  # parent class of a class is a class
        elif item.get(INSTANCE_OF):
            classes.update(item.get(INSTANCE_OF))
    print('found {} classes'.format(len(list(classes))))
    return classes


def get_classes(reduced_items: Iterable[dict], class_ids: Set[str])->Iterable[dict]:
    return filter(lambda i: i.get(ID) in class_ids, reduced_items)


def write_classes(input_dump: str, output: str):
        class_ids = get_class_ids(utils.get_json_dicts(input_dump))
        classes = get_classes(utils.get_json_dicts(input_dump), class_ids)
        utils.batch_write(map(lambda c: json.dumps(c), classes), output, config.BATCH_SIZE)


def main():
    write_classes(config.REDUCED_JSON_DUMP_PATH, config.REDUCED_CLASSES_JSON_DUMP_PATH)

if __name__ == '__main__':
    main()
