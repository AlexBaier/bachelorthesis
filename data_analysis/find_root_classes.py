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


# TODO: Iterable is flawed, because it is not lazy. Make it lazy.
# TODO: Does this iterable actually return classes?
def get_classes(reduced_items: typing.Iterable[dict])->typing.Iterable[dict]:
    """
    :param reduced_items:
    :return:
    """
    classes = set()  # type: typing.Set[ReducedItemWrapper]
    for item in reduced_items:
        if not item.get('P31') and item.get('P279'):
            classes.add(ReducedItemWrapper(item))
            classes.update(map(lambda i: ReducedItemWrapper(i), item.get('P279')))  # parent class of a class is a class
        elif item.get('P31'):
            classes.update(map(lambda i: ReducedItemWrapper(i), item.get('P31')))
    return map(lambda w: w.item)


def get_root_classes(classes: typing.Iterable[dict])->typing.Iterable[dict]:
    return filter(lambda c: not c.get('P31') and not c.get('P279'), classes)


def write_root_classes(input_dump: str, output: str):
        root_classes = map(lambda c: json.dumps(c),
                           get_root_classes(
                               get_classes(
                                   utils.get_json_dicts(input_dump))))

        utils.batch_write(root_classes, output, BATCH_SIZE)


def main():
    write_root_classes(REDUCED_JSON_DUMP_PATH, REDUCED_ROOT_CLASSES_JSON_DUMP_PATH)

if __name__ == '__main__':
    main()
