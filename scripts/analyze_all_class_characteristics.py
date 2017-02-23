import data_analysis.dumpio as dumpio
from data_analysis.class_extraction import analyze_characteristics


def main():
    class_characteristics_dump_path = '../data/class_characteristics-20161107'
    class_analysis_path = '../data/class_analysis-20161107'
    dumpio.JSONDumpWriter(class_analysis_path).write(
        [analyze_characteristics(dumpio.JSONDumpReader(class_characteristics_dump_path))])


if __name__ == '__main__':
    main()
