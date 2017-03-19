import json
import logging

from algorithm.sentence_gen import SentenceIterator
import algorithm.skipgram_model as sgm


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    classes_path = '../data/algorithm_io/class_ids-20161107.txt'
    training_paths = ['../data/algorithm_io/simple_sentences-20161107.txt',
                      '../data/algorithm_io/graph_walk_sentences-20161107-1.txt',
                      '../data/algorithm_io/graph_walk_sentences-20161107-2.txt',
                      '../data/algorithm_io/graph_walk_sentences-20161107-3.txt']
    output_path = '../data/algorithm_io/graph_walk_sentence_model-20161107'
    config_path = '../algorithm_config.json'

    model_config = 'graph walk sentence sgns'

    with open(config_path) as f:
        config = json.load(f)[model_config]

    sentences = SentenceIterator(training_paths)
    cids = list(map(lambda l: l.strip(), open(classes_path).readlines()))
    sgm.train_and_store_sgns(config=config, required_words=cids, sentences=sentences, output_path=output_path)


if __name__ == '__main__':
    main()
