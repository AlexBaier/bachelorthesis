import json
import logging

from evaluation.execution import execute_combined_algorithms


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    config_path = 'algorithm_config.json'

    training_data_path = paths_config['training data']
    test_data_path = paths_config['test data']
    output_path = paths_config['execution results']

    col_sep = ','
    row_sep = '\n'

    algorithms = ['ts+distknn(k=5)', 'ts+distknn(k=10)', 'ts+distknn(k=15)', 'ts+distknn(k=20)',
                  'ts+linproj', 'ts+pwlinproj(c=25)', 'ts+pwlinproj(c=50)',
                  'gw+distknn(k=5)', 'gw+distknn(k=10)', 'gw+distknn(k=15)', 'gw+distknn(k=20)',
                  'gw+linproj', 'gw+pwlinproj(c=25)', 'gw+pwlinproj(c=50)']

    results = execute_combined_algorithms(
        combined_algorithms=algorithms,
        config_path=config_path,
        training_data_path=training_data_path,
        test_data_path=test_data_path,
        workers=3
    )

    for algorithm in algorithms:
        computed_output_path = output_path.format(algorithm)
        with open(computed_output_path, mode='w') as f:
            for test_input, output in results[algorithm]:
                f.write('{}{}{}{}'.format(test_input, col_sep, output, row_sep))
        logging.log(level=logging.INFO,
                    msg='stored results for {} in {}'.format(algorithm, computed_output_path))

if __name__ == '__main__':
    main()
