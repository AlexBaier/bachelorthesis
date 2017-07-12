# Refinement of ontological taxonomies using a neural network approach
## Bachelor thesis

### Disclaimer
Source code, plots and other documents are licensed under CC 2.0 due to university requirements.

Written at University Koblenz-Landau at the Institute for Web Science and Technologies (WeST).

Python source code written in Python 3.5. Requirements for execution in **requirements.txt**.

### Structure

* **thesis/** contains the Bachelor thesis itself, and corresponding LaTeX files, as well as plots. 
  Compiling the document requires the [WeSTthesis](https://github.com/Institute-Web-Science-and-Technologies/WeSTthesis) class.
  For some figures, [draw.io](https://www.draw.io/) xml files are provided. Plots were generated via scripts or Jupyter notebooks,
  which can be found in other modules.
* **proposal/** contains the original proposal, which was required to start work on the Bachelor thesis. It is not relevant,
  since the thesis is finished.
* **algorithm/** is a Python module implementing the 3 distinct components of the developed hybrid algorithm,
  as described in Chapter "Hybrid Algorithm".
  Additional utility function for training input preparation are also provided.
* **data_analysis/** is a Python module implementing functionality for extracting and analyzing classes, relevant classes, 
  and orphan classes from a Wikidata JSON dump (dumps can be found [here](https://dumps.wikimedia.org/wikidatawiki/)).
  Results are discussed in Chapter "Taxonomy analysis".
* **evaluation/** is a Python module, which contains multiple important functionalities.
  In **evaluation.execution** an environment for the execution of hybrid algorithms is implemented.
  **evaluation.data_sample** contains the code functionality to generate disjoint sets of training and test data.
  **evaluation.statistics** implements the evaluation measures, accuracy and taxonomic overlap.
  The Jupyter notebook **evaluation/plots.ipynb** contains code to analyze and plot the performance of evaluated hybrid algorithms.
* **scripts/** contains scripts for the execution of aforementioned modules. Scripts use the configurations provided
  **paths_config.json**
* **irrelevant_properties.txt** contains the list of properties, which identify irrelevant classes. Explanation for this list
  in Chapter "Taxonomy analysis".
* **paths_config.json** defines all path constants used in scripts and the Jupyter notebook.
  It is recommended to only change the path strings, but never the field names, as such changes could affect multiple scripts.
* **algorithm_config.json** defines different types of components and hybrid algorithms, which are combinations of such components.
  Creating new components and hybrid algorithms is easily enabled via this configuration file.
 


 
  

