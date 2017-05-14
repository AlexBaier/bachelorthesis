import json
import logging

import algorithm.skipgram_model as sgm


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = 'graph walk sentence sgns'
    sequence_gen = 'graph walk sentences'

    with open('algorithm_config.json') as f:
        model_config = json.load(f)[model]

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    classes_path = paths_config['relevant class ids']
    sentences_path = paths_config[sequence_gen]
    output_path = paths_config[model]

    with open(classes_path) as f:
        cids = [l.strip() for l in f]

    sgm.train_and_store_sgns(config=model_config, required_words=cids, sentence_path=sentences_path,
                             output_path=output_path)
    logging.log(level=logging.INFO, msg='stored {} to {}'.format(model, output_path))


if __name__ == '__main__':
    main()
