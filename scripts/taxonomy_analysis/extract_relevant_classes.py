import json
import logging

from data_analysis.dumpio import JSONDumpReader, JSONDumpWriter


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open('paths_config.json') as f:
        paths_config = json.load(f)

    with open(paths_config['irrelevant properties']) as f:
        irrelevant_properties = set([l.strip() for l in f])

    classes_path = paths_config['class dump']
    relevant_classes_path = paths_config['relevant class dump']

    irrelevant_class_ids = set(filter(lambda c: not set(c['claims'].keys()).isdisjoint(irrelevant_properties),
                                      JSONDumpReader(classes_path)))
    logging.log(level=logging.INFO, msg='found {} irrelevant classes'.format(len(irrelevant_class_ids)))

    def relevant_classes():
        for rc in filter(lambda c: c['id'] not in irrelevant_class_ids, JSONDumpReader(classes_path)):
            for pid in rc['claims'].keys():
                for idx in range(len(rc['claims'][pid])):
                    # if the statement points to an irrelevant class, remove the statement
                    t = rc['claims'][pid]['mainsnak'].get('datavalue', dict()).get('value', dict()).get('numeric-id', 0)
                    if 'Q' + str(t) in irrelevant_class_ids:
                        rc['claims'][pid][idx] = None
                        logging.log(level=logging.INFO, msg='found irrelevant statement in {}-{}'.format(rc['id'], pid))
            yield rc

    JSONDumpWriter(relevant_classes_path).write(relevant_classes())
    logging.log(level=logging.INFO, msg='wrote relevant classes to {}'.format(relevant_classes_path))

if __name__ == '__main__':
    main()
