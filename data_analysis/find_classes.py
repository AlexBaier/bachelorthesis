"""
this module uses the output of reduce_json_dump.
"""
import json
import typing

import data_analysis.config as config
from data_analysis.constants import INSTANCE_OF, SUBCLASS_OF
import data_analysis.utils as utils


def get_class_ids(reduced_items: typing.Iterable[dict])->typing.Set[str]:
    """
    :param reduced_items:
    :return:
    """
    classes = set()  # type: typing.Set[str]
    for idx, item in enumerate(reduced_items):
        if not item.get(INSTANCE_OF) and item.get(SUBCLASS_OF):
            classes.add(item.get('id'))
            classes.update(item.get(SUBCLASS_OF))  # parent class of a class is a class
        elif item.get(INSTANCE_OF):
            classes.update(item.get(INSTANCE_OF))
        if idx % config.BATCH_SIZE == 0:
            print('found {} classes in {} items'.format(len(classes), idx))
    return classes


def get_classes(reduced_items: typing.Iterable[dict], class_ids: typing.Set[str])->typing.Iterable[dict]:
    return filter(lambda i: i.get('id') in class_ids, reduced_items)


def write_classes(input_dump: str, output: str):
        class_ids = get_class_ids(utils.get_json_dicts(input_dump))
        classes = map(lambda c: json.dumps(c),
                      get_classes(utils.get_json_dicts(input_dump), class_ids))

        utils.batch_write(classes, output, config.BATCH_SIZE)


def main():
    write_classes(config.REDUCED_JSON_DUMP_PATH, config.REDUCED_CLASSES_JSON_DUMP_PATH)

if __name__ == '__main__':
    main()
