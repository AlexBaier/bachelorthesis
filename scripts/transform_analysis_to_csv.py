import csv
import os

import data_analysis.dumpio as dumpio


def main():
    path = '../data'
    analysis_file_names = [
        'unlinked_labeled_instantiated_analysis-20161107',
        'unlinked_analysis-20161107'
    ]
    output_paths = list(map(lambda f: path + os.sep + f + '.csv', analysis_file_names))
    for idx, file_name in enumerate(analysis_file_names):
        analysis = list(dumpio.JSONDumpReader(path + os.sep + file_name))[0]
        with open(output_paths[idx], mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['count', analysis['unlinked class count']])
            count_types = ['property counts', 'instance counts', 'subclass counts']
            for count_type in count_types:
                writer.writerow([count_type])
                sorted_keys = list(sorted(map(int, analysis[count_type].keys())))
                writer.writerow([key for key in sorted_keys])
                writer.writerow([analysis[count_type][str(key)] for key in sorted_keys])
            freq_types = ['property frequencies']
            for freq_type in freq_types:
                writer.writerow([freq_type])
                sorted_freqs = list(sorted(analysis[freq_type].items(), key=lambda d: (int(d[1]), d[0]), reverse=True))
                writer.writerow([k for k, _ in sorted_freqs])
                writer.writerow([v for _, v in sorted_freqs])

if __name__ == '__main__':
    main()
