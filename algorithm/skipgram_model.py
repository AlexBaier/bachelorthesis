import gensim
from typing import Iterable, List


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


def train_and_store_sgns(config: dict, required_words: List[str], sentences: Iterable[List[str]], output_path: str):
    rule = SpecificWordTrim(include=required_words, exclude=list()).get_rule()
    model = gensim.models.Word2Vec(
        sentences=sentences,
        sg=1,
        size=config['embedding size'],
        window=config['window size'],
        alpha=config['initial learning rate'],
        min_count=config['min count'],
        max_vocab_size=config['max vocab size'],
        sample=config['subsampling'],
        hs=0,
        negative=config['negative sampling'],
        iter=config['iterations'],
        trim_rule=rule
    )
    model.delete_temporary_training_data()
    model.save(output_path)
