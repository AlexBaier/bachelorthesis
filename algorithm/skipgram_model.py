from typing import Iterator, List

import gensim


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


# TODO: functions shouldn't do 2 tasks
def train_and_store_sgns(config: dict, required_words: List[str], sentences: Iterator[List[str]], output_path: str):
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
    model.save(output_path)
