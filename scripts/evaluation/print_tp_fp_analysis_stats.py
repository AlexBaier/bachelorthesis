import json


def main():
    with open('paths_config.json') as f:
        paths_config = json.load(f)

    analysis_names = ['tp class analysis', 'fp class analysis']

    analysis = dict()
    for name in analysis_names:
        with open(paths_config[name]) as f:
            analysis[name] = json.load(f)

    for name in analysis_names:
        print("analysis: {}".format(name))
        print("total: {}".format(analysis[name]['class count']))
        print("labeled: {}".format(analysis[name]['labeled class count']))
        print("enwiki: {}".format(analysis[name]['enwiki count']))
        print("property average: {}".format(analysis[name]['property count average']))
        print("subclass average: {}".format(analysis[name]['subclass count average']))
        print("instance average: {}".format(analysis[name]['instance count average']))
        print("property median: {}".format(analysis[name]['property count median']))
        print("subclass median: {}".format(analysis[name]['subclass count median']))
        print("instance median: {}".format(analysis[name]['instance count median']))
        print()


if __name__ == '__main__':
    main()
