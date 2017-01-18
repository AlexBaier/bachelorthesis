import logging
import time

import gensim

import data_analysis.utils as utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class WikidataSentences(object):

    def __init__(self, file_path, sentence_length=3):
        self._file_path = file_path
        self._sentence_len = sentence_length

    def __iter__(self):
        with open(self._file_path) as f:
            sentence = list()
            for idx, l in enumerate(f):
                sentence.append(l.strip())
                if idx % self._sentence_len == (self._sentence_len - 1):
                    yield sentence
                    sentence = list()


class SpecificWordTrim(object):

    def __init__(self, include, exclude):
        self._include = set(include)
        self._exclude = set(exclude)
        assert len(self._include.intersection(self._exclude)) == 0

    def get_rule(self):
        def rule(word, count, min_count):
            if word in self._include:
                return gensim.models.utils.RULE_KEEP
            if word in self._exclude:
                return gensim.models.utils.RULE_DISCARD
            return gensim.models.utils.RULE_DEFAULT
        return rule


def main():
    start_time = time.strftime("%Y%m%d-%H%M%S")
    sentences = WikidataSentences('algorithm/data/text.txt', 3)
    class_ids = list(map(lambda j: j['id'], utils.get_json_dicts('/home/alex/PycharmProjects/thesis/data_analysis/output/reduced_classes')))
    logging.log(level=logging.INFO, msg='number of classes: {}'.format(len(class_ids)))
    rule = SpecificWordTrim(include=class_ids, exclude=list()).get_rule()
    # Configuration partially based on Levy2015.
    model = gensim.models.Word2Vec(
        sentences=sentences,
        sg=1,                   # skip-gram
        size=300,               # embedding size
        window=2,               # context window
        alpha=0.025,            # initial learning rate
        min_count=2,            # minimum number of word occurrences
        max_vocab_size=3e6,     # limited vocabulary size => approx 5GB memory usage
        sample=1e-05,           # threshold for down-sampling higher-frequency words
        hs=0,                   # use downsampling
        negative=20,            # noise words, try 20
        iter=5,                 # iterations over training corpus
        trim_rule=rule          # include all classes
    )
    logging.log(level=logging.INFO, msg='Completed training model.')
    model.delete_temporary_training_data(replace_word_vectors_with_normalized=True)
    logging.log(level=logging.INFO, msg='Attempting to store model.')
    model.save('algorithm/data/standard_model_' + start_time)

if __name__ == '__main__':
    main()
