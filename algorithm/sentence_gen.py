import abc
import logging
import sqlite3
from typing import Iterable, Iterator, List


class Wikidata2Sequence(object, metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def get_sequences(self)->Iterable[List[str]]:
        pass


class TripleSentences(Wikidata2Sequence):

    def __init__(self, items: Iterator[dict]):
        self.__items = items  # type: Iterator[dict]

    def get_sequences(self)->Iterable[List[str]]:
        def __get_sentences():
            for item in self.__items:
                item_id = item['id']
                for pid in item['claims'].keys():
                    stmt_group = item['claims'][pid]
                    for stmt in stmt_group:
                        if stmt['mainsnak']['snaktype'] == 'value' and stmt['mainsnak'].get('datatype'):
                            if stmt['mainsnak']['datatype'] == 'wikibase-item':
                                value = 'Q' + str(stmt['mainsnak']['datavalue']['value']['numeric-id'])
                            elif stmt['mainsnak']['datatype'] == 'wikibase-property':
                                value = 'P' + str(stmt['mainsnak']['datavalue']['value']['numeric-id'])
                            else:
                                value = stmt['mainsnak']['datatype']
                            yield [item_id, pid, value]
        return __get_sentences()


class GraphWalkSentences(Wikidata2Sequence):
    """
    GraphWalk has exponential runtime.
    """
    def __init__(self, vertices: List[str], depth: int, edge_store_path: str):
        self.__vertices = vertices  # type: List[str]
        self.__depth = depth  # type: int
        self.__edge_store = sqlite3.connect(edge_store_path)

    def get_sequences(self)->Iterable[List[str]]:
        def __get_sequences():
            l = len(self.__vertices)
            for idx, v in enumerate(self.__vertices):
                active_paths_from_v = self.__get_out_edges(v)
                finished_paths_from_v = list()
                current_depth = 1
                while current_depth <= self.__depth:
                    if len(active_paths_from_v) == 0:
                        break
                    current_path = active_paths_from_v.pop(0)
                    current_node = current_path[-1]
                    out_edges = self.__get_out_edges(current_node)
                    if not out_edges:
                        logging.log(level=logging.INFO, msg='{} has no'
                                                            ' out edges'.format(current_node))
                        finished_paths_from_v.append(current_path)
                    else:
                        for edge in out_edges:
                            new_path = current_path.copy()
                            new_path.append(edge[1])
                            new_path.append(edge[2])
                            active_paths_from_v.append(new_path)
                    current_depth += 1
                finished_paths_from_v.extend(active_paths_from_v)
                logging.log(level=logging.INFO, msg='{} paths discovered from {}'.format(len(finished_paths_from_v), v))
                for path in finished_paths_from_v:
                    yield path
                logging.log(level=logging.INFO, msg='graph walk progress: {}/{}'.format(idx+1, l))
        return __get_sequences()

    def __get_out_edges(self, v: str)->List[List[str]]:
        c = self.__edge_store.cursor()
        node = int(v[1:])
        c.execute('SELECT * FROM edges WHERE s=?', [node])
        results = c.fetchall()
        return list(map(lambda e: ['Q'+str(e[0]), 'P'+str(e[1]), 'Q'+str(e[2])], results))
