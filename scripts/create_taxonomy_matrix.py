from taxonomy.matrix import DownwardsTaxonomyGraph


def main():
    DownwardsTaxonomyGraph.write_graph_matrix('../data/classes-20161107', '../data/taxonomy_matrix-20161107')


if __name__ == '__main__':
    main()

