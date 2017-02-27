from typing import Iterable


def get_english_label(entity: dict)->str:
    """
    Gets the English label of a Wikidata entity, if it exists.
    Otherwise the empty string is returned.
    :param entity: Wikidata entity as dict.
    :return: English label of entity.
    """
    return entity.get('labels').get('en', dict()).get('value', '')


def get_wiki_title(entity: dict, wiki: str)->str:
    """
    Get the article title of the provided Wikidata entity of the specified Wikipedia.
    If the entity has no article on the specified Wikipedia, or there is no such Wikipedia,
    the empty string is returned.
    :param entity:
    :param wiki:
    :return: Wikipedia title of entity.
    """
    return entity.get('sitelinks', dict()).get(wiki, dict()).get('title', '')


def get_instance_of_ids(entity: dict)->Iterable[str]:
    """
    Returns an Iterable containing all IDs of the 'instance of' property of the supplied Wikidata entity.
    :param entity: Wikidata entity as dict.
    :return: Iterable containing IDs.
    """
    return get_item_property_ids('P31', entity)


def get_subclass_of_ids(entity: dict)->Iterable[str]:
    """
    Returns an Iterable containing all IDs of the 'subclass of' property of the supplied Wikidata entity.
    :param entity: Wikidata entity as dict.
    :return: Iterable containing IDs.
    """
    return get_item_property_ids('P279', entity)


def get_item_property_ids(property_id: str, entity: dict)->Iterable[str]:
    return map(lambda i: 'Q' + str(i),
               map(lambda e: e.get('mainsnak').get('datavalue').get('value').get('numeric-id'),
                   filter(lambda e: e.get('mainsnak').get('snaktype') == 'value',
                          entity.get('claims').get(property_id, list()))))

