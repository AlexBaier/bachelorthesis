import json
from typing import Dict, Iterable, Set


import data_analysis.config as config
from data_analysis.constants import INSTANCE_OF, SUBCLASS_OF, TOPICS_MAIN_CATEGORY
import data_analysis.utils as utils


def get_root_class_ids(root_classes: Iterable[dict])->Iterable[str]:
    return map(lambda e: e.get('id'), root_classes)


def get_children(root_class_ids: Iterable[str], items: Iterable[dict])->Dict[str, Dict[str, Set[str]]]:
    root_class_ids = set(root_class_ids)
    instances = dict()
    subclasses = dict()
    for item in items:
        for instance_of in item.get(INSTANCE_OF):
            if instance_of in root_class_ids:
                if not instances.get(instance_of, None):
                    instances[instance_of] = set()
                    instances[instance_of].add(item.get('id'))
                else:
                    instances[instance_of].add(item.get('id'))
        for subclass_of in item.get(SUBCLASS_OF):
            if subclass_of in root_class_ids:
                if not subclasses.get(subclass_of, None):
                    subclasses[subclass_of] = set()
                    subclasses[subclass_of].add(item.get('id'))
                else:
                    subclasses[subclass_of].add(item.get('id'))
    return {'subclasses': subclasses, 'instances': instances}


def get_characteristics(root_classes: Iterable[dict], root_class_ids: Iterable[str], items: Iterable[dict])\
        ->Iterable[str]:
    children = get_children(root_class_ids, items)
    subclasses = children.get('subclasses')
    instances = children.get('instances')
    for root_class in root_classes:
        result = dict()
        result['id'] = root_class.get('id')
        result['label'] = utils.get_english_label(root_class)
        result[TOPICS_MAIN_CATEGORY] = list(utils.get_item_property_ids(TOPICS_MAIN_CATEGORY, root_class))
        result['properties'] = list(root_class.get('claims').keys())
        result['subclasses'] = list(subclasses.get(result.get('id'), set()))
        result['instances'] = list(instances.get(result.get('id'), set()))
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
