import itertools
import json

import data_analysis.utils as utils
from data_analysis.constants import INSTANCE_OF, SUBCLASS_OF
import data_analysis.config as config


def extract_taxonomic_relations(entity: dict)->dict:
    result = dict()
    result['id'] = entity.get('id')
    result['label'] = utils.get_english_label(entity)
    result[INSTANCE_OF] = list(utils.get_instance_of_ids(entity))
    result[SUBCLASS_OF] = list(utils.get_subclass_of_ids(entity))

    return result


def write_reduced_json_lines(input_dump: str, output: str, limit: int=-1):
    reduced_json_strings = \
        map(lambda e: json.dumps(e),
            map(extract_taxonomic_relations,
                filter(utils.is_item,
                       utils.get_json_dicts(input_dump))))

    if limit >= 0:
        reduced_json_strings = itertools.islice(reduced_json_strings, limit)

    utils.file_write(reduced_json_strings, output)


def main(output: str, limit: int=-1):
    write_reduced_json_lines(config.JSON_DUMP_PATH, output, limit)

if __name__ == '__main__':
    main('reduced_items')
