import json
from typing import Dict, Iterable

import data_analysis.config as config
from data_analysis.constants import INSTANCES, PROPERTIES, SUBCLASSES, TOPICS_MAIN_CATEGORY
import data_analysis.utils as utils


def analyze_characteristics(characteristics: Iterable[dict])->dict:
    result = dict()
    root_class_count = 0
    property_counts = dict()  # type: Dict[int, int]
    subclass_counts = dict()  # type: Dict[int, int]
    instance_counts = dict()  # type: Dict[int, int]
    property_frequencies = dict() # type: Dict[str, int]
    topic_frequencies = dict()  # type: Dict[str, int]
    for ch in characteristics:
        property_num = len(ch.get(PROPERTIES))
        subclass_num = len(ch.get(SUBCLASSES))
        instance_num = len(ch.get(INSTANCES))
        # count number of analyzed root classes
        root_class_count += 1
        # count number of root classes with a specific number of properties
        property_counts[property_num] = property_counts.get(property_num, 0) + 1
        # count number of root classes with a specific number of subclasses
        subclass_counts[subclass_num] = subclass_counts.get(subclass_num, 0) + 1
        # count number of root classes with a specific number of instances
        instance_counts[instance_num] = instance_counts.get(instance_num, 0) + 1
        # count frequency of properties in root classes
        for prop in ch.get(PROPERTIES):
            property_frequencies[prop] = property_frequencies.get(prop, 0) + 1
        # count frequency of topics in root classes
        for topic in ch.get(TOPICS_MAIN_CATEGORY):
            topic_frequencies[topic] = topic_frequencies.get(topic, 0) + 1

    result['root class count'] = root_class_count
    result['property counts'] = property_counts
    result['subclass counts'] = subclass_counts
    result['instance counts'] = instance_counts
    result['property_frequencies'] = property_frequencies
    result['topic frequencies'] = topic_frequencies

    return result


def write_analysis(characteristics: str, output: str):
    characteristics = utils.get_json_dicts(characteristics)
    analysis = analyze_characteristics(characteristics)
    utils.batch_write([json.dumps(analysis)], output, 1)


def main():
    write_analysis(config.ROOT_CLASS_CHARACTERISTICS_PATH, config.ROOT_CLASS_ANALYSIS_PATH)


if __name__ == '__main__':
    main()
