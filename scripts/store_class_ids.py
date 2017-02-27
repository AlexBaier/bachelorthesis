from data_analysis.dumpio import JSONDumpReader


def main():
    class_dump_path = '../data/classes-20161107'
    class_id_path = '../data/algorithm_io/class_ids-20161107.txt'

    with open(class_id_path, mode='w') as f:
        for c in JSONDumpReader(class_dump_path):
            f.write(c['id'] + '\n')


if __name__ == '__main__':
    main()

