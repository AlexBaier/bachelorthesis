"""
TODO: Rework it so that it writes to a file, instead of STDOUT.
"""
import itertools
import json
from typing import Iterable, Tuple

import data_analysis.utils as utils
from data_analysis.config import JSON_DUMP_PATH


def is_parentless_item(entity: dict)->bool:
    return utils.is_item(entity) and is_not_instance(entity) and is_not_subclass(entity)


def is_not_instance(entity: dict)->bool:
    return entity.get('claims') .get('P279', None) is None


def is_not_subclass(entity: dict)->bool:
    return entity.get('claims').get('P31', None) is None


def get_parentless_items(entities: Iterable[dict])->Iterable[dict]:
    return filter(is_parentless_item, entities)


def get_id_label_pairs(entities: Iterable[dict])->Iterable[Tuple[str, str]]:
    return map(get_id_label_pair, entities)


def get_id_label_pair(entity: dict)->Tuple[str, str]:
    return entity.get('id'), utils.get_english_label(entity)


def id_label_pair_to_csv(pair: Tuple[str, str])->str:
    return '{},{}'.format(pair[0], pair[1])


def dict_to_json_string(entity: dict)->str:
    return json.dumps(entity)


def print_ids_and_labels_of_all_parentless_items(file_path: str, limit: int=-1)->None:
    id_label_csv_rows = \
        map(id_label_pair_to_csv, get_id_label_pairs(
            get_parentless_items(
                utils.get_json_dicts(file_path))))

    if limit >= 0:
        id_label_csv_rows = itertools.islice(id_label_csv_rows, limit)

    for row in id_label_csv_rows:
        print(row)


def print_json_of_all_parentless_items(file_path: str, limit: int=-1)->None:
    json_strings = \
        map(dict_to_json_string,
            get_parentless_items(
                utils.get_json_dicts(file_path)))

    if limit >= 0:
        json_strings = itertools.islice(json_strings, limit)

    for row in json_strings:
        print(row)


def main():
    print_json_of_all_parentless_items(JSON_DUMP_PATH)


if __name__ == '__main__':
    main()
