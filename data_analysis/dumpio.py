import json
import logging

import typing


class JSONDumpReader(typing.Iterator[dict]):

    def __init__(self, dump_path: str):
        self.__dump_path = dump_path

    def __iter__(self):
        with open(self.__dump_path) as f:
            for l in f:
                l = JSONDumpReader.__clean_line(l)
                try:
                    yield json.loads(l)
                except ValueError:
                    logging.log(level=logging.DEBUG, msg="encountered illegal string while parsing JSON dump")

    @staticmethod
    def __clean_line(l: str)->str:
        return l.strip().rstrip(',')


class JSONDumpWriter(object):

    def __init__(self, output_path: str, batch_size: int=500):
        self.__output_path = output_path
        self.__batch_size = batch_size

    def write(self, objects: typing.Iterable[dict]):
        with open(self.__output_path, mode='w') as f:
            f.write('[\n')
            batch = list()
            for idx, o in enumerate(objects):
                batch.append(json.dumps(o))
                if idx and idx % self.__batch_size == 0:
                    logging.log(level=logging.INFO, msg='wrote {} objects'.format(idx + 1))
                    f.write('\n'.join(batch) + '\n')
                    batch = list()
            f.write('\n'.join(batch) + '\n')
            f.write(']\n')
