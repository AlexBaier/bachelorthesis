import json
import typing

import data_analysis.config as config
from data_analysis.constants import INSTANCE_OF, SUBCLASS_OF
import data_analysis.utils as utils


def is_parentless(reduced_class: dict)->bool:
    return not reduced_class.get(INSTANCE_OF) and not reduced_class.get(SUBCLASS_OF)


def get_root_class_ids(reduced_classes: typing.Iterable[dict])->typing.Iterable[str]:
    return map(lambda c: c.get('id'),
               filter(is_parentless, reduced_classes))


def get_root_classes(entities: typing.Iterable[dict], root_class_ids: typing.Iterable)->typing.Iterable[dict]:
    root_class_ids = set(root_class_ids)
    return filter(lambda e: e.get('id') in root_class_ids, entities)


def write_root_classes(json_dump: str, reduced_classes_dump: str, output: str):
    root_class_ids = get_root_class_ids(utils.get_json_dicts(reduced_classes_dump))
    root_classes = map(lambda c: json.dumps(c),
                       get_root_classes(utils.get_json_dicts(json_dump), root_class_ids))

    utils.batch_write(root_classes, output, config.BATCH_SIZE)


def main():
    write_root_classes(config.JSON_DUMP_PATH, config.REDUCED_CLASSES_JSON_DUMP_PATH, config.ROOT_CLASSES_JSON_DUMP_PATH)


if __name__ == '__main__':
    main()
