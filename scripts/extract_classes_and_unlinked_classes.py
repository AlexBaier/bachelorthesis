import logging

import data_analysis.dumpio as dumpio
import data_analysis.pipe as pipe
import data_analysis.class_extraction as extraction

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main():
    dump_path = ''
    output_path = ''
    class_ids = extraction.get_class_ids(dumpio.JSONDumpReader(dump_path))
    is_class = lambda e: e['id'] in class_ids
    pipe.DataPipe.read_json_dump(dump_path).filter_by(extraction.is_item).filter_by(is_class).write(output_path)


if __name__ == '__main__':
    main()
