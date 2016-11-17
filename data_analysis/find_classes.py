"""
this module uses the output of reduce_json_dump.
"""
import json
import typing

from data_analysis.config import REDUCED_JSON_DUMP_PATH, REDUCED_ROOT_CLASSES_JSON_DUMP_PATH
import data_analysis.utils as utils

BATCH_SIZE = 500


class ReducedItemWrapper(object):
    """
    Used for set operations.
    """
    def __init__(self, item: dict):
        self.item = item
        self.id = item.get('id')

    def __hash__(self):
        return int(self.id[1:])

    def __eq__(self, other):
        if isinstance(other, ReducedItemWrapper):
            return self.id == other.id
        return False


def get_class_ids(reduced_items: typing.Iterable[dict])->typing.Set[str]:
    """
    :param reduced_items:
    :return:
    """
    classes = set()  # type: typing.Set[str]
    for idx, item in enumerate(reduced_items):
        if not item.get('P31') and item.get('P279'):
            classes.add(item.get('id'))
            classes.update(item.get('P279'))  # parent class of a class is a class
        elif item.get('P31'):
            classes.update(item.get('P31'))
        if idx % BATCH_SIZE == 0:
            print('found {} classes in {} items'.format(len(classes), idx))
    return classes


def get_classes(reduced_items: typing.Iterable[dict], class_ids: typing.Set[str])->typing.Iterable[dict]:
    return filter(lambda i: i.get('id') in class_ids, reduced_items)


def write_classes(input_dump: str, output: str):
        class_ids = get_class_ids(utils.get_json_dicts(input_dump))
        classes = map(lambda c: json.dumps(c),
                      get_classes(utils.get_json_dicts(input_dump), class_ids))

        utils.batch_write(classes, output, BATCH_SIZE)


def main():
    write_classes(REDUCED_JSON_DUMP_PATH, REDUCED_ROOT_CLASSES_JSON_DUMP_PATH)

if __name__ == '__main__':
    main()
