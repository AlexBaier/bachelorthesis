import os

import algorithm.skipgram_model as sgm
import data_analysis.utils as utils


def main():
    rel_path = '../data'
    model_file_name = 'standard_model_20170117-163852'
    output_path = rel_path + os.sep + 'sim_matrix_' + model_file_name
    cids = list(map(lambda j: j['id'], utils.get_json_dicts('../data_analysis/output/reduced_classes')))
    m = sgm.TrainedStoredModel(rel_path + os.sep + model_file_name)
    m.write_similarity_matrix(words=cids, output_path=output_path)


if __name__ == '__main__':
    main()
