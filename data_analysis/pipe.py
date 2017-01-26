"""
This is probably not a good idea. The chances are high that this is the opposite of the pythonic way.
And there is probably a library, which does it better out there anyway.
"""
from typing import Callable, Iterable, List, Set

import data_analysis.dumpio as dumpio


class DataPipe(object):

    def __init__(self, content: Iterable):
        self.__content = content  # type: Iterable

    @staticmethod
    def read_json_dump(dump_path: str)->'DataPipe':
        return DataPipe(dumpio.JSONDumpReader(dump_path))

    @staticmethod
    def iterate(iterable: Iterable)->'DataPipe':
        return DataPipe(iterable)

    def filter_by(self, rule: Callable[[object], bool])->'DataPipe':
        return DataPipe(filter(rule, self.__content))

    def map(self, mapping: Callable)->'DataPipe':
        return DataPipe(map(mapping, self.__content))

    def to_list(self)-> List:
        return list(self.__content)

    def to_iterable(self)->Iterable:
        return self.__content

    def to_set(self)->Set:
        return set(self.__content)

    def write(self, output_path: str):
        dumpio.JSONDumpWriter(output_path).write(self.__content)
