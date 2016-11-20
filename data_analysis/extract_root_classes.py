"""
extract_and_analyse.py is a script, which will execute all steps needed to find and analyze all root classes
of the Wikidata JSON dump, defined in data_analysis.config. Multiple files are created in the process. Make sure to have
enough disk space free and set the paths in data_analysis.config to point to the right files.
"""
import data_analysis.find_characteristics as find_characteristics
import data_analysis.find_classes as find_classes
import data_analysis.find_root_classes as find_root_classes
import data_analysis.reduce_json_dump as reduce_json_dump


def main():
    print('start reducing JSON dump')
    reduce_json_dump.main()
    print('finished reducing JSON dump')
    print('start finding classes')
    find_classes.main()
    print('finished finding classes')
    print('start finding root classes')
    find_root_classes.main()
    print('finished finding root classes')
    print('start finding root classes characteristics')
    find_characteristics.main()
    print('finished finding root classes characteristics')


if __name__ == '__main__':
    main()

