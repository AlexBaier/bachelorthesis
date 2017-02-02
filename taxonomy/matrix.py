from typing import List, Dict

import data_analysis.dumpio as dumpio
import data_analysis.utils as utils


class DownwardsTaxonomyGraph(object):
    """
    Represents a directed graph of a taxonomy.
    Each edge represents the is-parent-of relationship, which implies for n1->n2, that n1 is the parent
    or super class of n2.
    Therefore the graph is called downwards, as the direction of the edges points from the top-level nodes
    to the low-level nodes.
    """

    def __init__(self, matrix_path: str):
        self.__matrix = list(dumpio.JSONDumpReader(matrix_path))[0]

    def get_direct_subclasses(self, class_id: str)->List[str]:
        return self.__matrix[class_id]

    def breadth_first_search(self, class_id: str)->Dict[str, int]:
        """
        Executes a breadth first search over the taxonomy graph.
        Output is the distance of each class in regards to the entered class.
        If there is no walk between the entered class and and a target class, the distance is -1.
        :param class_id:
        :return: dict with Wikidata ID of nodes as key, and distance as value.
        """
        result = dict((k, -1) for k in self.__matrix.keys())
        visited = set()
        to_visit = list()
        not_visited = lambda c: c not in visited
        to_visit.append((class_id, 0))
        while len(to_visit) > 0:
            current, layer = to_visit.pop(0)
            if current in visited:
                continue
            result[current] = layer
            visited.add(current)
            to_visit.extend(map(lambda c: (c, layer+1), filter(not_visited, self.get_direct_subclasses(current))))
        return result

    @staticmethod
    def write_graph_matrix(classes_path: str, output_path: str):
        subclasses = dict()
        for c in dumpio.JSONDumpReader(classes_path):
            if not subclasses.get(c['id'], None):
                subclasses[c['id']] = list()
            for cid in utils.get_subclass_of_ids(c):
                if not subclasses.get(cid, None):
                    subclasses[cid] = list()
                subclasses[cid].append(c['id'])

        dumpio.JSONDumpWriter(output_path).write([subclasses])
