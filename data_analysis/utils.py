import json
import typing

from data_analysis.constants import INSTANCE_OF, SUBCLASS_OF


def get_json_dicts(file_path: str)->typing.Iterable[dict]:
    """
    :param file_path: file_path to a JSON dump, which has only 1 object in each line.
    :return: Iterable containing dicts of Wikidata entities.
    """
    with open(file_path) as f:
        for line in f:
            line = clean_line(line)
            if line and is_not_square_bracket(line):
                yield json.loads(line)


def clean_line(s: str)->str:
    return s.strip().rstrip(',')


def is_not_square_bracket(s: str)->bool:
    return s not in ['[', ']']


def is_item(entity: dict)->bool:
    """
    :param entity: dict of Wikidata entity.
    :return: true if entity is an item, else false.
    """
    return entity.get('id')[0] == 'Q'


def get_english_label(entity: dict)->str:
    """
    Gets the English label of a Wikidata entity, if it exists.
    Otherwise the empty string is returned.
    :param entity: Wikidata entity as dict.
    :return: English label of entity.
    """
    return entity.get('labels').get('en', {}).get('value', '')


def get_instance_of_ids(entity: dict)->typing.Iterable[str]:
    """
    Returns an Iterable containing all IDs of the 'instance of' property of the supplied Wikidata entity.
    :param entity: Wikidata entity as dict.
    :return: Iterable containing IDs.
    """
    return get_item_property_ids(INSTANCE_OF, entity)


def get_subclass_of_ids(entity: dict)->typing.Iterable[str]:
    """
    Returns an Iterable containing all IDs of the 'subclass of' property of the supplied Wikidata entity.
    :param entity: Wikidata entity as dict.
    :return: Iterable containing IDs.
    """
    return get_item_property_ids(SUBCLASS_OF, entity)


def get_item_property_ids(property_id: str, entity: dict)->typing.Iterable[str]:
    return map(lambda i: 'Q' + str(i),
               map(lambda e: e.get('mainsnak').get('datavalue').get('value').get('numeric-id'),
                   filter(lambda e: e.get('mainsnak').get('snaktype') == 'value',
                          entity.get('claims').get(property_id, list()))))


def batch_write(lines: typing.Iterable[str], output: str, batch_size: int):
    """
    Writes lines in batches to the supplied output file path.
    Writing in batches seems to be faster than writing each line separately.
    I could be wrong.
    :param lines: Iterable with strings. Strings should not contain the line break \n.
    :param output: file path, to which this function writes.
    :param batch_size: Number of lines, which get written to the file in one operation.
    :return: None.
    """
    with open(output, mode='w') as f:
        batch = list()
        print('begin writing to {}'.format(output))
        for idx, l in enumerate(lines):
            batch.append(l)
            if idx > 0 and idx % batch_size == 0:
                f.write('\n'.join(batch) + '\n')
                batch = list()
                print('{} items already written'.format(idx))
        print('done')
