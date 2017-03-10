import random
from typing import Iterable, List, TypeVar, Generic, Tuple

import data_analysis.utils as utils

T = TypeVar('T')


class MultiLabelSample(Generic[T]):

    def __init__(self, input_arg: T, possible_outputs: List[T]):
        self.input_arg = input_arg  # type: T
        self.possible_outputs = possible_outputs  # type: List[T]

    def __str__(self):
        return '{}->{}'.format(self.input_arg, self.possible_outputs)

    def to_csv_row(self, col_sep: str=',', row_sep: str='\n')->str:
        return col_sep.join([str(self.input_arg)] + list(map(str, self.possible_outputs))) + row_sep


def generate_wikidata_classification_samples(classes: Iterable[dict], property_id: str)->\
        Iterable[MultiLabelSample[str]]:
    return filter(lambda s: s.possible_outputs,
                  map(lambda c: MultiLabelSample(c['id'], list(utils.get_item_property_ids(property_id, c))), classes))


def generate_training_test_data(samples: Iterable[MultiLabelSample[str]], test_samples: int)->\
        Tuple[List[MultiLabelSample[str]], List[MultiLabelSample[str]]]:
    random.seed(42)
    samples = list(samples)
    random.shuffle(samples)
    return samples[:test_samples], samples[test_samples:]
