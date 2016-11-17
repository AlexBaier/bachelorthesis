import json
import typing


def get_json_dicts(file_path: str)->typing.Iterable[dict]:
    """
    :param file_path: file_path to a Wikidata JSON dump
    :return: iterator returning dicts of Wikidata entities
    """
    with open(file_path) as f:
        for line in f:
            line = clean_line(line)
            # Fix for two JSON objects in one line.
            idx = line.find('}{')
            if idx != -1:  # TODO: Fix issue in reduce_json_dump.py.
                yield json.loads(line[0:idx+1])
                yield json.loads(line[idx+1:])
            elif is_not_square_bracket(line):
                yield json.loads(line)


def clean_line(s: str)->str:
    return s.strip().rstrip(',')


def is_not_square_bracket(s: str)->bool:
    return s not in ['[', ']']


def is_item(entity: dict)->bool:
    """
    :param entity: dict of Wikidata entity
    :return: true if entity is an item, else false
    """
    return entity.get('id')[0] == 'Q'


def get_english_label(entity: dict)->str:
    return entity.get('labels').get('en', {}).get('value', '')


def get_instance_of_ids(entity: dict)->typing.Iterable[str]:
    return get_item_property_ids('P31', entity)


def get_subclass_of_ids(entity: dict)->typing.Iterable[str]:
    return get_item_property_ids('P279', entity)


def get_item_property_ids(property_id: str, entity: dict):
    return map(lambda i: 'Q' + str(i),
               map(lambda e: e.get('mainsnak').get('datavalue').get('value').get('numeric-id'),
                   filter(lambda e: e.get('mainsnak').get('snaktype') == 'value',
                          entity.get('claims').get(property_id, list()))))


def batch_write(values: typing.Iterable[str], output: str, batch_size: int):
    batch = list()
    with open(output, mode='w') as f:
        print('begin writing to {}'.format(output))
        for idx, l in enumerate(values):
            batch.append(l)
            if idx % batch_size == 0:
                f.write('\n'.join(batch))
                batch = list()
                print('{} items already written'.format(idx))
        f.write('\n'.join(batch))
        print('done')
