import data_analysis.dumpio as dumpio
from data_analysis.class_extraction import analyze_characteristics


def main():
    unlinked_characteristics_dump_path = '../data/unlinked_characteristics-20161107'
    unlinked_analysis_path = '../data/unlinked_analysis-20161107'
    dumpio.JSONDumpWriter(unlinked_analysis_path).write(
        [analyze_characteristics(dumpio.JSONDumpReader(unlinked_characteristics_dump_path))])


if __name__ == '__main__':
    main()
