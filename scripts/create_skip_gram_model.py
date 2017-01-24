import time

import algorithm.skipgram_model as sgm
import data_analysis.dumpio as dumpio


def main():
    training_path = '../data/text.txt'
    output_path = '../data/standard_model_' + time.strftime("%Y%m%d-%H%M%S")
    config = {
        'size': 300, # embedding size
        'window': 2,  # context window
        'alpha': 0.025,  # initial learning rate
        'min_count': 2,  # minimum number of word occurrences
        'max_vocab_size': 3e6,  # limited vocabulary size => approx 5GB memory usage
        'sample': 1e-05,  # threshold for down-sampling higher-frequency words
        'negative': 20,  # noise words, try 20
        'iter': 5  # iterations over training corpus
    }
    cids = list(map(lambda j: j['id'], dumpio.JSONDumpReader('../data_analysis/output/reduced_classes')))
    sgm.train_and_store_skip_gram_model(config=config, required_words=cids,
                                        training_path=training_path, output_path=output_path)


if __name__ == '__main__':
    main()
