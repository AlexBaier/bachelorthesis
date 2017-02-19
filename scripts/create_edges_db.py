import logging

import sqlite3
from typing import Iterator, List


class EdgeIterator(Iterator[List[str]]):

    def __init__(self, triple_sentence_dump: str):
        self.__dump = triple_sentence_dump

    def __iter__(self):
        with open(self.__dump) as f:
            for l in f:
                yield l.strip().split()


def generate_edge_store(db_path: str, edges: Iterator[List[str]]):
    logging.log(level=logging.INFO, msg='start generating edge store')
    conn = sqlite3.connect(db_path)
    with conn:
        c = conn.cursor()
        c.execute('DROP TABLE IF EXISTS edges')
        c.execute('CREATE TABLE edges (s int, r int, t int)')
        for idx, edge in enumerate(filter(lambda e: e[2][0] == 'Q', edges)):
            t = [int(edge[0][1:]), int(edge[1][1:]), int(edge[2][1:])]
            c.execute('INSERT INTO edges VALUES(?, ?, ?)', t)
            if idx % 10000 == 0:
                logging.log(level=logging.INFO, msg='processed {} edges'.format(idx + 1))
    logging.log(level=logging.INFO, msg='finished generating edge store')


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    edge_store_path = '../data/algorithm_io/edges.sqlite3'
    triple_sentence_path = '../data/algorithm_io/simple_sentences-20161107.txt'
    generate_edge_store(edge_store_path, EdgeIterator(triple_sentence_path))


if __name__ == '__main__':
    main()
