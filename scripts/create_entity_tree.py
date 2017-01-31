from data_analysis import dumpio
from taxonomy.matrix import DownwardsTaxonomyGraph


def main():
    g = DownwardsTaxonomyGraph('../data/taxonomy_matrix-20161107')
    dumpio.JSONDumpWriter('../data/entity_tree-20161107').write([g.breadth_first_search('Q35120')])


if __name__ == '__main__':
    main()
