import data_analysis.pipe as pipe
import data_analysis.class_extraction as ce


def main():
    entity_dump_path = '../data/wikidata/wikidata-20161107-all.json'
    unlinked_class_dump_path = '../data/unlinked_classes-20161107'
    unlinked_characteristic_dump_path = '../data/unlinked_characteristics-20161107'
    get_id = lambda c: c['id']
    unlinked_class_ids = pipe.DataPipe.read_json_dump(unlinked_class_dump_path).map(get_id).to_set()
    entities = pipe.DataPipe.read_json_dump(entity_dump_path).to_iterable()
    to_characteristic = ce.to_characteristic(unlinked_class_ids, entities)
    to_dict = lambda ch: ch.to_dict()
    pipe.DataPipe.read_json_dump(unlinked_class_dump_path).map(to_characteristic).map(to_dict)\
        .write(unlinked_characteristic_dump_path)

if __name__ == '__main__':
    main()
