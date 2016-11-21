import json
from typing import Dict, Iterable, Set


import data_analysis.config as config
from data_analysis.constants import INSTANCE_OF, SUBCLASS_OF, \
    TOPICS_MAIN_CATEGORY, SUBCLASSES, INSTANCES, PROPERTIES, ID, LABEL, CLAIMS
import data_analysis.utils as utils


def get_root_class_ids(root_classes: Iterable[dict])->Iterable[str]:
    return map(lambda e: e.get(ID), root_classes)


def get_children(root_class_ids: Iterable[str], items: Iterable[dict])->Dict[str, Dict[str, Set[str]]]:
    root_class_ids = set(root_class_ids)
    instances = dict()
    subclasses = dict()
    for item in items:
        for instance_of in item.get(INSTANCE_OF):
            if instance_of in root_class_ids:
                if not instances.get(instance_of, None):
                    instances[instance_of] = set()
                    instances[instance_of].add(item.get(ID))
                else:
                    instances[instance_of].add(item.get(ID))
        for subclass_of in item.get(SUBCLASS_OF):
            if subclass_of in root_class_ids:
                if not subclasses.get(subclass_of, None):
                    subclasses[subclass_of] = set()
                    subclasses[subclass_of].add(item.get(ID))
                else:
                    subclasses[subclass_of].add(item.get(ID))
    return {SUBCLASSES: subclasses, INSTANCES: instances}


def get_characteristics(root_classes: Iterable[dict], root_class_ids: Iterable[str], items: Iterable[dict])\
        ->Iterable[str]:
    children = get_children(root_class_ids, items)
    subclasses = children.get(SUBCLASSES)
    instances = children.get(INSTANCES)
    for root_class in root_classes:
        result = dict()
        result[ID] = root_class.get(ID)
        result['enwiki'] = utils.get_enwiki_title(root_class)
        result[LABEL] = utils.get_english_label(root_class)
        result[TOPICS_MAIN_CATEGORY] = list(utils.get_item_property_ids(TOPICS_MAIN_CATEGORY, root_class))
        result[PROPERTIES] = list(root_class.get(CLAIMS).keys())
        result[SUBCLASSES] = list(subclasses.get(result.get(ID), set()))
        result[INSTANCES] = list(instances.get(result.get(ID), set()))
        yield json.dumps(result)


def write_characteristics(root_classes: str, reduced_items: str, output: str):
    items = utils.get_json_dicts(reduced_items)
    classes = utils.get_json_dicts(root_classes)
    class_ids = get_root_class_ids(utils.get_json_dicts(root_classes))

    characteristics = get_characteristics(classes, class_ids, items)

    utils.batch_write(characteristics, output, config.BATCH_SIZE)


def main():
    write_characteristics(
        config.ROOT_CLASSES_JSON_DUMP_PATH,
        config.REDUCED_JSON_DUMP_PATH,
        config.ROOT_CLASS_CHARACTERISTICS_PATH)


if __name__ == '__main__':
    main()
