import logging
from typing import List

import gensim

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


class TrainedStoredModel(object):

    def __init__(self, model_path: str):
        self.__model = gensim.models.Word2Vec.load(model_path)  # type: gensim.models.Word2Vec

    def write_similarity_matrix(self, words: List[str], output_path: str, sparse: bool=False):
        if not sparse:
            with open(output_path, mode='w') as f:
                for word1 in words:
                    row = list()
                    for word2 in words:
                        row.append(str(self.__model.similarity(word1, word2).item()))
                    f.write(','.join(row) + '\n')
        else:
            # Remember which pairs where already computed.
            computed_pairs = set()
            with open(output_path, mode='w') as f:
                for word1 in words:
                    for word2 in words:
                        pass


# TODO: functions shouldn't do 2 tasks
def train_and_store_skip_gram_model(config: dict, required_words: List[str], training_path: str, output_path: str):
    sentences = WikidataSentences(training_path, 3)
    rule = SpecificWordTrim(include=required_words, exclude=list()).get_rule()
    model = gensim.models.Word2Vec(
        sentences=sentences,
        sg=1,
        size=config['size'],
        window=config['window'],
        alpha=config['alpha'],
        min_count=config['min_count'],
        max_vocab_size=config['max_vocab_size'],
        sample=config['sample'],
        hs=0,
        negative=config['negative'],
        iter=config['iter'],
        trim_rule=rule
    )
    model.delete_temporary_training_data(replace_word_vectors_with_normalized=True)
    model.save(output_path)
