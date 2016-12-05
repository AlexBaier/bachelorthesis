import json

import data_analysis.config as config
from data_analysis.analyze_characteristics import analyze_characteristics
import data_analysis.utils as utils


def main():
    characteristics = filter(lambda c: c['label'], utils.get_json_dicts(config.ROOT_CLASS_CHARACTERISTICS_PATH))
    analysis = analyze_characteristics(characteristics)
    utils.batch_write([json.dumps(analysis)], config.LABELED_ROOT_CLASS_ANALYSIS_PATH, 1)

if __name__ == '__main__':
    main()
