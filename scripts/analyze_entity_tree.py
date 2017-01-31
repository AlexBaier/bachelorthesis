from data_analysis import dumpio


def main():
    id2layer = list(dumpio.JSONDumpReader('../data/entity_tree-20161107'))[0]
    hist = dict()
    class_count = 0
    for cid in id2layer.keys():
        if not hist.get(id2layer[cid], None):
            hist[id2layer[cid]] = 0
        hist[id2layer[cid]] += 1
        class_count += 1
    print('contained classes:', class_count - hist[-1])
    print('not contained classes:', hist[-1])

    perc_hist = dict()
    for k, v in hist.items():
        perc_hist[k] = float(v)/float(class_count)
    for l in perc_hist.items():
        print(l)
    for l in hist.items():
        print(l)

if __name__ == '__main__':
    main()
