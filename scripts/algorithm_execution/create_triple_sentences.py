import json
import logging

from algorithm.sentence_gen import TripleSentences
from data_analysis.dumpio import JSONDumpReader


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        config = json.load(f)

    dump_path = config['wikidata dump']
    output_path = config['triple sentences']

    sentences = TripleSentences(JSONDumpReader(dump_path)).get_sequences()
    with open(output_path, mode='w') as f:
        for idx, sentence in enumerate(map(lambda s: ' '.join(s) + '\n', sentences)):
            f.write(sentence)
            if idx % 10000 == 0:
                logging.log(level=logging.INFO, msg='wrote {} sentences'.format(idx + 1))
    logging.log(level=logging.INFO, msg='wrote triple sentences to {}'.format(output_path))

if __name__ == '__main__':
    main()
