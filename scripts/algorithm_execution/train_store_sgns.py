import json
import logging

import algorithm.skipgram_model as sgm
from algorithm.sentence_gen import SentenceIterable


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = 'triple sentence sgns'
    sequence_gen = 'triple sentences'

    with open('algorithm_config.json') as f:
        model_config = json.load(f)[model]

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    classes_path = paths_config['class ids']
    sentences_path = paths_config[sequence_gen]
    output_path = paths_config[model]

    sentences = SentenceIterable(sentences_path)

    cids = list(map(lambda l: l.strip(), open(classes_path).readlines()))
    sgm.train_and_store_sgns(config=model_config, required_words=cids, sentences=sentences, output_path=output_path)


if __name__ == '__main__':
    main()
