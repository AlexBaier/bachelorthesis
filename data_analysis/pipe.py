import typing

import data_analysis.dumpio as dumpio


class DataPipe(object):

    def __init__(self, content: typing.Iterable):
        self.__content = content  # type: typing.Iterable

    @staticmethod
    def read_json_dump(dump_path: str)->'DataPipe':
        return DataPipe(dumpio.JSONDumpReader(dump_path))

    @staticmethod
    def iterate(iterable: typing.Iterable)->'DataPipe':
        return DataPipe(iterable)

    def filter_by(self, rule)->'DataPipe':
        return DataPipe(filter(rule, self.__content))

    def map(self, mapping)->'DataPipe':
        return DataPipe(map(mapping, self.__content))

    def to_list(self)->typing.List:
        return list(self.__content)

    def to_iterable(self)->typing.Iterable:
        return self.__content

    def to_set(self)->typing.Set:
        return set(self.__content)

    def write(self, output_path: str):
        dumpio.JSONDumpWriter(output_path).write(self.__content)
