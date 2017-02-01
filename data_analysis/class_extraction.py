from typing import Callable, Dict, Iterable, List, Set, Tuple

import data_analysis.utils as utils


class ClassCharacteristic(object):

    def __init__(self, cid: str, label: str, enwiki: str,
                 properties: List[str], instances: List[str], subclasses: List[str]):
        self.id = cid
        self.label = label
        self.enwiki = enwiki
        self.properties = properties
        self.instances = instances
        self.subclasses = subclasses

    def to_dict(self):
        return {
            'id': self.id,
            'label': self.label,
            'enwiki': self.enwiki,
            'properties': self.properties,
            'instances': self.instances,
            'subclasses': self.subclasses
        }


def get_class_ids(entities: Iterable[dict])->Set[str]:
    """
    :param entities:
    :return:
    """
    classes = set()
    for e in entities:
        subclass_of = list(utils.get_subclass_of_ids(e))
        instance_of = list(utils.get_instance_of_ids(e))
        if subclass_of:
            classes.add(e['id'])
            classes.update(subclass_of)
        if instance_of:
            classes.update(instance_of)
    return classes


def is_item(entity: dict)->bool:
    """
    :param entity: dict of Wikidata entity.
    :return: true if entity is an item, else false.
    """
    return entity.get('id')[0] == 'Q'


def is_unlinked_class(c: dict)->bool:
    """
    :param c:
    :return:
    """
    return not list(utils.get_subclass_of_ids(c))


def get_class_children(entities: Iterable, class_ids: Set[str])->Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    :param entities:
    :param class_ids: Set of class IDs, whose children should be retrieved.
        If class_ids is empty, the children all classes will be returned.
    :return: (instances, subclasses)
    """
    instances = dict()
    subclasses = dict()
    for e in entities:
        for instance_of in utils.get_instance_of_ids(e):
            if not class_ids or instance_of in class_ids:
                if not instances.get(instance_of, None):
                    instances[instance_of] = list()
                instances[instance_of].append(e['id'])
        for subclass_of in utils.get_subclass_of_ids(e):
            if not class_ids or subclass_of in class_ids:
                if not subclasses.get(subclass_of, None):
                    subclasses[subclass_of] = list()
                subclasses[subclass_of].append(e['id'])
    return instances, subclasses


def to_characteristic(class_ids: Set[str], entities: Iterable[dict])->Callable[[dict], ClassCharacteristic]:
    """
    :param class_ids:
    :param entities:
    :return:
    """
    instances, subclasses = get_class_children(entities, class_ids)

    def __to_characteristic(c: dict)->ClassCharacteristic:
        return ClassCharacteristic(
            cid=c['id'],
            label=utils.get_english_label(c),
            enwiki=utils.get_wiki_title(c, 'enwiki'),
            properties=list(c['claims'].keys()),
            instances=instances.get(c['id'], list()),
            subclasses=subclasses.get(c['id'], list())
        )

    return __to_characteristic


def analyze_characteristics(characteristics: Iterable[dict])->dict:
    result = dict()
    unlinked_class_count = 0
    enwiki_count = 0
    labeled_count = 0
    property_counts = dict()  # type: Dict[int, int]
    subclass_counts = dict()  # type: Dict[int, int]
    instance_counts = dict()  # type: Dict[int, int]
    property_frequencies = dict()  # type: Dict[str, int]
    for ch in characteristics:
        property_num = len(ch['properties'])
        subclass_num = len(ch['subclasses'])
        instance_num = len(ch['instances'])
        # count number of analyzed classes
        unlinked_class_count += 1
        # count number of classes with enwiki
        enwiki_count += 1 if ch['enwiki'] else 0
        # count number of labeled classes
        labeled_count += 1 if ch['label'] else 0
        # count number of root classes with a specific number of properties
        property_counts[property_num] = property_counts.get(property_num, 0) + 1
        # count number of root classes with a specific number of subclasses
        subclass_counts[subclass_num] = subclass_counts.get(subclass_num, 0) + 1
        # count number of root classes with a specific number of instances
        instance_counts[instance_num] = instance_counts.get(instance_num, 0) + 1
        # count frequency of properties in root classes
        for prop in ch['properties']:
            property_frequencies[prop] = property_frequencies.get(prop, 0) + 1

    result['unlinked class count'] = unlinked_class_count
    result['enwiki count'] = enwiki_count
    result['labeled class count'] = labeled_count
    result['property counts'] = property_counts
    result['subclass counts'] = subclass_counts
    result['instance counts'] = instance_counts
    result['property frequencies'] = property_frequencies

    return result
