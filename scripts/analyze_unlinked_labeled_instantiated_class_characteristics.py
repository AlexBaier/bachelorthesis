from data_analysis.class_extraction import analyze_characteristics
import data_analysis.dumpio as dumpio
import data_analysis.pipe as pipe


def main():
    characteristics_dump_path = '../data/unlinked_characteristics-20161107'
    analysis_path = '../data/unlinked_labeled_instantiated_analysis-20161107'
    is_labeled = lambda c: c['label']
    has_instances = lambda c: c['instances']
    dumpio.JSONDumpWriter(analysis_path).write(
        [analyze_characteristics(
            # Filter characteristics to only contain classes with labels and instances.
            pipe.DataPipe.read_json_dump(characteristics_dump_path)
                .filter_by(is_labeled)
                .filter_by(has_instances)
                .to_iterable())])


if __name__ == '__main__':
    main()
