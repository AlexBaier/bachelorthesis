from evaluation.execution import execute_combined_algorithms


def main():
    config_path = '../algorithm_config.json'
    training_data_path = '../evaluation/training_data-20161107.csv'
    test_data_path = '../evaluation/test_data-20161107.csv'
    output_path = '../evaluation/results_{}-20161107.csv'

    col_sep = ','
    row_sep = '\n'

    combined_algorithms = {'distknn'}

    results = execute_combined_algorithms(
        combined_algorithms=combined_algorithms,
        config_path=config_path,
        training_data_path=training_data_path,
        test_data_path=test_data_path
    )

    for combined_algorithm in combined_algorithms:
        with open(output_path.format(combined_algorithm), mode='w') as f:
            for test_input, output in results[combined_algorithm]:
                f.write('{}{}{}{}'.format(test_input, col_sep, output, row_sep))


if __name__ == '__main__':
    main()
