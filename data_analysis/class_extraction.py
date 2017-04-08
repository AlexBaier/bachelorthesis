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


def is_orphan_class(c: dict)->bool:
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
    class_count = 0
    enwiki_count = 0
    labeled_count = 0

    property_count_hist = dict()  # type: Dict[int, int]
    subclass_count_hist = dict()  # type: Dict[int, int]
    instance_count_hist = dict()  # type: Dict[int, int]

    property_hist = dict()  # type: Dict[str, int]
    subclass_hist = dict()  # type: Dict[str, int]
    instance_hist = dict()  # type: Dict[str, int]

    for ch in characteristics:
        class_count += 1
        enwiki_count += 1 if ch['enwiki'] else 0
        labeled_count += 1 if ch['label'] else 0

        property_num = len(ch['properties'])
        subclass_num = len(ch['subclasses'])
        instance_num = len(ch['instances'])
        property_count_hist[property_num] = property_count_hist.get(property_num, 0) + 1
        subclass_count_hist[subclass_num] = subclass_count_hist.get(subclass_num, 0) + 1
        instance_count_hist[instance_num] = instance_count_hist.get(instance_num, 0) + 1

        for prop in ch['properties']:
            property_hist[prop] = property_hist.get(prop, 0) + 1
        for inst in ch['instances']:
            instance_hist[inst] = instance_hist.get(inst, 0) + 1
        for subc in ch['subclasses']:
            subclass_hist[subc] = subclass_hist.get(subc, 0) + 1

    property_count_avg = utils.average(list(property_count_hist.keys()), list(property_count_hist.values()))
    subclass_count_avg = utils.average(list(subclass_count_hist.keys()), list(subclass_count_hist.values()))
    instance_count_avg = utils.average(list(instance_count_hist.keys()), list(instance_count_hist.values()))

    property_count_median = utils.median(list(property_count_hist.keys()), list(property_count_hist.values()))
    subclass_count_median = utils.median(list(subclass_count_hist.keys()), list(subclass_count_hist.values()))
    instance_count_median = utils.median(list(instance_count_hist.keys()), list(instance_count_hist.values()))

    result = dict()

    result['class count'] = class_count
    result['enwiki count'] = enwiki_count
    result['labeled class count'] = labeled_count

    result['property count histogram'] = property_count_hist
    result['subclass count histogram'] = subclass_count_hist
    result['instance count histogram'] = instance_count_hist

    result['property histogram'] = property_hist
    result['subclass histogram'] = subclass_hist
    result['instance histogram'] = instance_hist

    result['property count average'] = property_count_avg
    result['subclass count average'] = subclass_count_avg
    result['instance count average'] = instance_count_avg

    result['property count median'] = property_count_median
    result['subclass count median'] = subclass_count_median
    result['instance count median'] = instance_count_median

    return result
