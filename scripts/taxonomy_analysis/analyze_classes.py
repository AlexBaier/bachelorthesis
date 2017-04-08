import json
import logging

from data_analysis.class_extraction import analyze_characteristics
from data_analysis.dumpio import JSONDumpReader


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        config = json.load(f)

    characteristics_path = config['class characteristics']
    analysis_path = config['class analysis']

    with open(analysis_path, mode='w') as f:
        json.dump(analyze_characteristics(JSONDumpReader(characteristics_path)), f)
    logging.log(level=logging.INFO, msg='wrote analysis to {}'.format(analysis_path))

if __name__ == '__main__':
    main()
