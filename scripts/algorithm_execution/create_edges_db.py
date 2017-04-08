import json
import logging
import sqlite3

from typing import Iterable, List


def generate_edge_store(db_path: str, edges: Iterable[List[str]]):
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

    with open('paths_config.json') as f:
        config = json.load(f)

    triple_sentence_path = config['triple sentences']
    edge_store_path = config['edges db']

    with open(triple_sentence_path) as f:
        edges = filter(lambda s: len(s) == 3, (l.strip().split() for l in f))
    generate_edge_store(edge_store_path, edges)
    logging.log(level=logging.INFO, msg='stored edge db to {}'.format(edge_store_path))


if __name__ == '__main__':
    main()
