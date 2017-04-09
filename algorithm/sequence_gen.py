import abc
import logging
import random
import time

import numpy as np
import pathos.multiprocessing as mp
from typing import Callable, Iterable, List, Set, Tuple


class SequenceGen(object, metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def get_sequences(self)->Iterable[List[str]]:
        pass


class TripleSentences(SequenceGen):

    def __init__(self, items: Iterable[dict], forbidden_properties: Set[str],
                 is_forbidden_triple: Callable[[List[str]], bool]):
        self.__forbidden_properties = forbidden_properties  # type: Set[str]
        self.__is_forbidden_triple = is_forbidden_triple  # type: Callable[[List[str]], bool]
        self.__items = items  # type: Iterable[dict]

    def get_sequences(self)->Iterable[List[str]]:
        def __get_sentences():
            for item in self.__items:
                item_id = item['id']
                # skip all items which contain forbidden properties
                if not set(item['claims'].keys()).isdisjoint(self.__forbidden_properties):
                    continue
                for pid in item['claims'].keys():
                    stmt_group = item['claims'][pid]
                    for stmt in stmt_group:
                        if stmt['mainsnak']['snaktype'] == 'value' and stmt['mainsnak'].get('datatype'):
                            if stmt['mainsnak']['datatype'] == 'wikibase-item':
                                value = 'Q' + str(stmt['mainsnak']['datavalue']['value']['numeric-id'])
                            elif stmt['mainsnak']['datatype'] == 'wikibase-property':
                                value = 'P' + str(stmt['mainsnak']['datavalue']['value']['numeric-id'])
                            else:
                                value = ''
                            sentence = [item_id, pid, value] if value else [item_id, pid]
                            # if sentence is triple and forbidden, don't generate it
                            if len(sentence) == 3 and self.__is_forbidden_triple(sentence):
                                continue
                            yield sentence
        return __get_sentences()


class GraphWalkSentences(SequenceGen):
    """
    GraphWalk has exponential runtime.
    """
    def __init__(self, vertices: List[str], depth: int, max_walks_per_v: int,
                 get_out_edges: Callable[[str], List[Tuple[str, str]]], workers: int=4):
        self.__vertices = vertices  # type: List[str]
        self.__depth = depth  # type: int
        self.__max_walks = max_walks_per_v  # type: int
        self.__get_out_edges = get_out_edges  # type: Callable[[str], List[Tuple[str, str]]]
        self.__workers = workers  # type: int

    def get_sequences(self)->Iterable[List[str]]:
        logging.log(level=logging.INFO, msg='Graph walk sequences with depth={}, max walks={}, workers={}'.format(
            self.__depth, self.__max_walks, self.__workers
        ))

        def __get_sequences():
            with mp.Pool(processes=self.__workers) as p:
                for walks in p.imap_unordered(self.__get_walks, self.__vertices):
                    for w in walks:
                        yield w

        return __get_sequences()

    def __get_walks(self, vertice: str):
        start_time = time.time()

        random.seed()

        walks = [[vertice if idx == 0 else '' for idx, _ in enumerate(range(2*self.__depth+1))]
                 for _ in range(self.__max_walks)]

        for current_depth in range(1, self.__depth):
            for current_walk in range(self.__max_walks):
                current_vertice = walks[current_walk][2*current_depth-2]
                # current walk already stopped
                if current_vertice == '':
                    continue
                try:
                    out_edges = self.__get_out_edges(current_vertice)
                except IndexError:
                    out_edges = list()
                m = len(out_edges)
                # current vertice has no out-edges => skip this vertice
                if m == 0:
                    continue
                if m == 1:
                    r = 0
                else:
                    r = np.random.randint(0, m-1, 1)[0]
                chosen_edge = out_edges[r]
                walks[current_walk][2*current_depth-1] = chosen_edge[1]  # add edge weight to walk
                walks[current_walk][2*current_depth] = chosen_edge[2]  # add target to walk

        # strip empty strings of walk
        for walk_id in range(self.__max_walks):
            walks[walk_id] = list(filter(lambda s: s != '', walks[walk_id]))
        walks = list(filter(lambda walk: len(walk) > 1, walks))

        duration = time.time() - start_time

        logging.log(level=logging.INFO, msg='{} walks from {} in {} seconds'.format(len(walks), vertice, duration))
        return walks


class Sequences(Iterable[List[str]]):

    def __init__(self, file_paths: List[str]):
        self.__paths = file_paths  # type: List[str]

    def __iter__(self):
        for path in self.__paths:
            with open(path) as f:
                for s in map(lambda l: l.strip().split(), f):
                    yield s
