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

    combined_algorithms = ['ts+pwlinproj(c=85)', 'ts+pwlinproj(c=100)', 'ts+pwlinproj(c=70)']

    results = execute_combined_algorithms(
        combined_algorithms=combined_algorithms,
        config_path=config_path,
        training_data_path=training_data_path,
        test_data_path=test_data_path,
        workers=1
    )

    for combined_algorithm in combined_algorithms:
        computed_output_path = output_path.format(combined_algorithm)
        with open(computed_output_path, mode='w') as f:
            for test_input, output in results[combined_algorithm]:
                f.write('{}{}{}{}'.format(test_input, col_sep, output, row_sep))
        logging.log(level=logging.INFO,
                    msg='stored results for {} in {}'.format(combined_algorithm, computed_output_path))

if __name__ == '__main__':
    main()
