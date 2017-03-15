# TODO: rename module
from typing import List

from evaluation.data_sample import MultiLabelSample
from evaluation.utils import load_test_data


class SingleLabelMultiClassClassifierEvaluation(object):

    def __init__(self, result_path: str, test_data_path: str, algorithm: str='unnamed', col_sep: str=','):
        self.__algorithm = algorithm
        with open(result_path) as f:
            self.__predictions = dict([(input_arg, prediction)
                                       for input_arg, prediction in map(lambda l: l.strip().split(col_sep), f)])
        self.__golds = load_test_data(test_data_path)  # type: List[MultiLabelSample]

    def get_total_count(self)->int:
        return len(self.__golds)

    def get_true_positive_count(self)->int:
        tp_count = 0
        for gold in self.__golds:
            if self.__predictions[gold.input_arg] in gold.possible_outputs:
                tp_count += 1
        return tp_count
