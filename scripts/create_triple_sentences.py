from algorithm.sentence_gen import TripleSentences
from data_analysis.dumpio import JSONDumpReader


def main():
    dump_path = '../data/wikidata/wikidata-20161107-all.json'
    output_path = '../data/algorithm_io/simple_sentences-20161107.txt'
    sentences = TripleSentences(JSONDumpReader(dump_path)).get_sequences()
    with open(output_path, mode='w') as f:
        c = 1
        for sentence in map(lambda s: ' '.join(s) + '\n', sentences):
            f.write(sentence)
            if c % 1000 == 0:
                print('written', str(c), 'sentences')
            c += 1

if __name__ == '__main__':
    main()
