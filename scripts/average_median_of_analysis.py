import csv
from typing import List


def map_to_int(l: List[str])->List[int]:
    return list(map(lambda s: int(s), l))


def average(x, y):
    return float(sum([a * b for a, b in zip(x, y)]))/sum(y)


def median(x, y):
    median_pos = sum(y)/2
    before = 0
    for a, b in zip(x, y):
        before += b
        if median_pos <= before:
            return a
    return x[-1]


def main():
    analysis_paths = [
        '../data/unlinked_analysis-20161107.csv',
        '../data/unlinked_labeled_instantiated_analysis-20161107.csv'
    ]
    for path in analysis_paths:
        with open(path) as f:
            for idx, row in enumerate(map(lambda l: l.split(','), f)):
                if idx == 0:
                    class_count = int(row[1])
                elif idx == 2:
                    property_count = map_to_int(row)
                elif idx == 3:
                    property_class_count = map_to_int(row)
                elif idx == 5:
                    instance_count = map_to_int(row)
                elif idx == 6:
                    instance_class_count = map_to_int(row)
                elif idx == 8:
                    subclass_count = map_to_int(row)
                elif idx == 9:
                    subclass_class_count = map_to_int(row)
            print('property avg:', average(property_count, property_class_count))
            print('property median:', median(property_count, property_class_count))
            print('instance avg:', average(instance_count, instance_class_count))
            print('instance median:', median(instance_count, instance_class_count))
            print('subclass avg:', average(subclass_count, subclass_class_count))
            print('subclass median:', median(subclass_count, subclass_class_count))


if __name__ == '__main__':
    main()
