"""
The similarity histograms show that for all algorithms, the same kind of misclassifications occurs.
In the prediction-gold similarity interval of [0.1, 0.4] a spike occurs in the histograms.
This script outputs a file commonalities of the unknowns which produce this effect.
"""
import logging

from data_analysis.pipe import DataPipe
from data_analysis.utils import get_english_label, get_subclass_of_ids, get_instance_of_ids
from evaluation.statistics import get_prediction_gold_cosine_similarities
from evaluation.utils import load_config, load_embeddings_and_labels, load_test_data


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    config_path = '../algorithm_config.json'
    test_data_path = '../evaluation/test_data-20161107.csv'
    results_path = '../evaluation/results_{}-20161107.csv'
    classes_path = '../data/classes-20161107'

    algorithm = 'ts+kriknn(k=15&r=1)'

    interval_begin = 0.1
    interval_end = 0.4

    max_length = 100

    output_path = '../evaluation/unknowns_analysis_{}_{}_{}-20161107.csv'.format(
        algorithm, str(interval_begin).replace('.', ''), str(interval_end).replace('.', ''))

    config = load_config(config_path)
    sgns = config['combinations'][algorithm]['sgns']
    logging.log(level=logging.INFO, msg='loaded algorithm config')

    embeddings, labels = load_embeddings_and_labels(config[sgns]['embeddings path'])
    id2idx = dict((label, idx) for idx, label in enumerate(labels))
    logging.log(level=logging.INFO, msg='loaded embeddings of {}'.format(sgns))

    with open(results_path.format(algorithm)) as f:
        predictions = dict((u, p) for u, p in map(lambda l: l.strip().split(','), f))
    logging.log(level=logging.INFO, msg='loaded predictions of {}'.format(algorithm))

    test_data = load_test_data(test_data_path)
    logging.log(level=logging.INFO, msg='loaded test data')

    similarities = get_prediction_gold_cosine_similarities(predictions, test_data,
                                                           lambda item_id: embeddings[id2idx[item_id]])
    logging.log(level=logging.INFO, msg='computed similarities of prediction-gold pairs')

    is_in_relevant_interval = (interval_begin <= similarities) * (similarities <= interval_end)
    relevant_unknown_ids = set()

    for idx in range(len(test_data)):
        if is_in_relevant_interval[idx]:
            relevant_unknown_ids.add(test_data[idx].input_arg)
    logging.log(level=logging.INFO, msg='found {} unknowns in interval [{}, {}]'.format(len(relevant_unknown_ids),
                                                                                        interval_begin, interval_end))

    def to_detail(obj):
        result = dict()
        result['properties'] = list(obj['claims'].keys())
        result['label'] = get_english_label(obj)
        result['instance of'] = list(get_instance_of_ids(obj))
        result['subclass of'] = list(get_subclass_of_ids(obj))
        return result

    details = DataPipe.read_json_dump(classes_path)\
        .filter_by(lambda obj: obj['id'] in relevant_unknown_ids)\
        .map(to_detail)\
        .to_list()
    logging.log(level=logging.INFO, msg='extracted details of unknowns in interval')

    labeled_count = 0
    instance_freq = dict()
    subclass_freq = dict()
    prop_freq = dict()
    for detail in details:
        labeled_count += 1 if detail['label'] else 0
        for subclass in detail['subclass of']:
            subclass_freq[subclass] = subclass_freq.setdefault(subclass, 0) + 1
        for instance in detail['instance of']:
            instance_freq[instance] = instance_freq.setdefault(instance, 0) + 1
        for prop in detail['properties']:
            prop_freq[prop] = prop_freq.setdefault(prop, 0) + 1
    logging.log(level=logging.INFO, msg='aggregated details')

    with open(output_path, mode='w') as f:
        f.write(algorithm + '\n')
        f.write(','.join(['interval', str(interval_begin), str(interval_end)]) + '\n')
        f.write(','.join(['total', str(len(relevant_unknown_ids))]) + '\n')
        f.write(','.join(['labeled', str(labeled_count)]) + '\n')
        i, c = zip(*sorted(instance_freq.items(), key=lambda o: o[1])[::-1][:max_length])
        f.write(','.join(i) + '\n')
        f.write(','.join(map(lambda o: str(o), c)) + '\n')
        i, c = zip(*sorted(subclass_freq.items(), key=lambda o: o[1])[::-1][:max_length])
        f.write(','.join(i) + '\n')
        f.write(','.join(map(lambda o: str(o), c)) + '\n')
        i, c = zip(*sorted(prop_freq.items(), key=lambda o: o[1])[::-1][:max_length])
        f.write(','.join(i) + '\n')
        f.write(','.join(map(lambda o: str(o), c)) + '\n')
    logging.log(level=logging.INFO, msg='wrote aggregated details')

if __name__ == '__main__':
    main()
