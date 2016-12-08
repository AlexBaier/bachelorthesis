"""
build_sparse_taxonomy_matrix.py reads the JSON dump in data_analysis.config.REDUCED_CLASSES_JSON_DUMP_PATH,
and creates a sparse matrix of the taxonomy. It outputs the sparse matrix as csv in
data_analysis.config.SPARSE_MATRIX_PATH, which has the format "subclass,superclass" as int ids,
and it outputs the mapping of int ids to Wikidata item ids as csv in data_analysis.config.CLASS_MAPPING_PATH,
which has the format "int,Wikdata ID".
"""

import json

import data_analysis.config as config


def main():
    classes = set()
    superclass_relations = list()
    with open(config.REDUCED_CLASSES_JSON_DUMP_PATH) as f:
        for l in f:
            if not l.strip() == '':
                j = json.loads(l)
                classes.add(j['id'])
                classes.update(j['P279'])
                superclass_relations.append((j['id'], j['P279']))
    # create a mapping of ints in [0, #classes-1] to Wikidata item IDs.
    class_mappings = dict()
    for i, c in enumerate(classes):
        class_mappings[c] = i
    sparse_matrix_csv = ''
    for c, r in superclass_relations:
        for s in r:
            sparse_matrix_csv += str(class_mappings[c]) + ',' + str(class_mappings[s]) + '\n'
    mapping_csv = '\n'.join([str(i) + ',' + c for c, i in class_mappings.items()])

    with open(config.SPARSE_MATRIX_PATH, mode='w') as f:
        f.write(sparse_matrix_csv)
    with open(config.CLASS_MAPPING_PATH, mode='w') as f:
        f.write(mapping_csv)


if __name__ == '__main__':
    main()
