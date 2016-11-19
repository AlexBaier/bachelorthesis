import itertools
import json

import data_analysis.utils as utils
from data_analysis.constants import INSTANCE_OF, SUBCLASS_OF, ID, LABEL
import data_analysis.config as config


def extract_taxonomic_relations(entity: dict)->dict:
    result = dict()
    result[ID] = entity.get(ID)
    result[LABEL] = utils.get_english_label(entity)
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

    utils.batch_write(reduced_json_strings, output, config.BATCH_SIZE)


def main(limit: int=-1):
    write_reduced_json_lines(config.JSON_DUMP_PATH, config.REDUCED_JSON_DUMP_PATH, limit)

if __name__ == '__main__':
    main()
