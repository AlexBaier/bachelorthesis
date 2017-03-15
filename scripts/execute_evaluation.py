from evaluation.evaluation import SingleLabelMultiClassClassifierEvaluation


def main():
    test_data_path = '../evaluation/test_data-20161107.csv'
    result_path = '../evaluation/results_{}-20161107.csv'

    output_path = '../evaluation/evaluation_{}-20161107.csv'

    col_sep = ','

    algorithms = ['baseline']

    for algorithm in algorithms:
        ev = SingleLabelMultiClassClassifierEvaluation(
            result_path.format(algorithm), test_data_path, algorithm=algorithm)
        total_count = ev.get_total_count()
        tp_count = ev.get_true_positive_count()

        with open(output_path.format(algorithm), mode='w') as f:
            f.write('algorithm{}{}\n'.format(col_sep, algorithm))
            f.write('total{}{}\n'.format(col_sep, total_count))
            f.write('tp{}{}\n'.format(col_sep, tp_count))
            f.write('tp-ratio{}{}\n'.format(col_sep, float(tp_count)/total_count))


if __name__ == '__main__':
    main()
