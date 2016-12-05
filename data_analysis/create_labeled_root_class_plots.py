import data_analysis.config as config
from data_analysis.create_plots import create_plots


def main():
    create_plots(config.LABELED_ROOT_CLASS_ANALYSIS_PATH, config.LABELED_PROPERTY_SUM_FIGURE_PATH,
                 config.LABELED_INSTANCE_SUM_FIGURE_PATH, config.LABELED_SUBCLASS_SUM_FIGURE_PATH,
                 config.LABELED_PROPERTY_FREQUENCY_FIGURE_PATH)


if __name__ == '__main__':
    main()
