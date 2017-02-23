import data_analysis.pipe as pipe
import data_analysis.class_extraction as ce


def main():
    entity_dump_path = '../data/wikidata/wikidata-20161107-all.json'
    class_dump_path = '../data/classes-20161107'
    class_characteristic_dump_path = '../data/class_characteristics-20161107'
    get_id = lambda c: c['id']
    class_ids = pipe.DataPipe.read_json_dump(class_dump_path).map(get_id).to_set()
    entities = pipe.DataPipe.read_json_dump(entity_dump_path).to_iterable()
    to_characteristic = ce.to_characteristic(class_ids, entities)
    to_dict = lambda ch: ch.to_dict()
    pipe.DataPipe.read_json_dump(class_dump_path).map(to_characteristic).map(to_dict)\
        .write(class_characteristic_dump_path)

if __name__ == '__main__':
    main()
