import logging
import time

from algorithm.sentence_gen import SentenceIterator
import algorithm.skipgram_model as sgm


def main():
    # 2017-03-01 07:49:56,481 : INFO : training on 1595945650 raw words
    #   (373764633 effective words) took 45693.4s, 8180 effective words/s
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    classes_path = '../data/algorithm_io/class_ids-20161107.txt'
    training_path = '../data/algorithm_io/simple_sentences-20161107.txt'
    output_path = '../data/algorithm_io/simple_sentence_model-20161107'
    config = {
        'size': 300,  # embedding size
        'window': 2,  # context window
        'alpha': 0.025,  # initial learning rate
        'min_count': 1,  # minimum number of word occurrences
        'max_vocab_size': 3e6,  # limited vocabulary size => approx 5GB memory usage
        'sample': 1e-05,  # threshold for down-sampling higher-frequency words
        'negative': 15,  # noise words, try 20
        'iter': 5  # iterations over training corpus
    }
    sentences = SentenceIterator(training_path)
    cids = list(map(lambda l: l.strip(), open(classes_path).readlines()))
    sgm.train_and_store_sgns(config=config, required_words=cids, sentences=sentences, output_path=output_path)


if __name__ == '__main__':
    main()
