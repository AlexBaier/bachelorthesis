import logging

import data_analysis.dumpio as dumpio
import evaluation.data_gen as data_gen


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    classes_path = '../data/classes-20161107'
    test_data_output = '../evaluation/test_data-20161107.csv'
    training_data_output = '../evaluation/training_data-20161107.csv'
    test_sample_count = 200000

    samples = data_gen.generate_wikidata_classification_samples(dumpio.JSONDumpReader(classes_path), property_id='P279')
    test_samples, training_samples = data_gen.generate_training_test_data(samples, test_sample_count)
    logging.log(level=logging.INFO,
                msg='test samples: {}, training samples: {}'.format(len(test_samples), len(training_samples)))

    with open(test_data_output, mode='w') as f:
        f.write(str(len(test_samples)) + '\n')
        for test_sample in test_samples:
            f.write(test_sample.to_csv_row(',', '\n'))

    with open(training_data_output, mode='w') as f:
        f.write(str(len(training_samples)) + '\n')
        for training_sample in training_samples:
            f.write(training_sample.to_csv_row(',', '\n'))


if __name__ == '__main__':
    main()
