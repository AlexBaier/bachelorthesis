import json
import logging

import pandas as pd

from data_analysis.class_extraction import analyze_characteristics
from data_analysis.dumpio import JSONDumpReader
from evaluation.utils import load_test_data


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    evaluation_path = paths_config['evaluation']

    results = pd.read_csv(evaluation_path, sep=';')
    algorithm = results[results['accuracy'] == results['accuracy'].max()].iloc[0]['algorithm']
    logging.info('analysis best performing algorithm {}'.format(algorithm))

    test_data_path = paths_config['test data']
    golds = dict((gold_sample.input_arg, set(gold_sample.possible_outputs))
                 for gold_sample in load_test_data(test_data_path))

    result_path = paths_config['execution results'].format(algorithm)
    tps = set()
    fps = set()
    with open(result_path) as f:
        for unknown, prediction in map(lambda l: l.strip().split(','), f):
            if prediction in golds[unknown]:
                tps.add(unknown)
            else:
                fps.add(unknown)

    characteristics_path = paths_config['class characteristics']
    tp_analysis_path = paths_config['tp class analysis']
    fp_analysis_path = paths_config['fp class analysis']

    with open(tp_analysis_path, mode='w') as f:
        json.dump(
            analyze_characteristics(
                filter(lambda c: c['id'] in tps, JSONDumpReader(characteristics_path))
            ), f)
    logging.log(level=logging.INFO, msg='wrote analysis to {}'.format(tp_analysis_path))

    with open(fp_analysis_path, mode='w') as f:
        json.dump(
            analyze_characteristics(
                filter(lambda c: c['id'] in fps, JSONDumpReader(characteristics_path))
            ), f)
    logging.log(level=logging.INFO, msg='wrote analysis to {}'.format(fp_analysis_path))


if __name__ == '__main__':
    main()
