import json
from typing import Iterable

import data_analysis.config as config
from data_analysis.constants import SUBCLASS_OF, ID
import data_analysis.utils as utils


def is_parentless(reduced_class: dict)->bool:
    """
    Returns true, if the class has no superclass, otherwise false.
    :param reduced_class:
    :return: boolean
    """
    return not reduced_class.get(SUBCLASS_OF)


def get_root_class_ids(reduced_classes: Iterable[dict])->Iterable[str]:
    root_class_ids = list()
    for idx, c in enumerate(reduced_classes):
        if is_parentless(c):
            root_class_ids.append(c.get(ID))
        if idx % 1000 == 0:
            print('iterated {} classes'.format(idx))
    return root_class_ids


def get_root_classes(entities: Iterable[dict], root_class_ids: Iterable)->Iterable[dict]:
    root_class_ids = set(root_class_ids)
    return filter(lambda e: e.get(ID) in root_class_ids, entities)


def write_root_classes(json_dump: str, reduced_classes_dump: str, output: str):
    root_class_ids = get_root_class_ids(utils.get_json_dicts(reduced_classes_dump))
    root_classes = map(lambda c: json.dumps(c),
                       get_root_classes(utils.get_json_dicts(json_dump), root_class_ids))
    utils.batch_write(root_classes, output, config.BATCH_SIZE)


def main():
    write_root_classes(config.JSON_DUMP_PATH, config.REDUCED_CLASSES_JSON_DUMP_PATH, config.ROOT_CLASSES_JSON_DUMP_PATH)


if __name__ == '__main__':
    main()
