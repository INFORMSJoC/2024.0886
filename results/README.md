## Organization

This folder is divided into three folders:
- paper_figures includes all the figures (with respective name) in the paper.
- family_setups_small_experiments_results contains the pickle files for each of the rows in the tables from Section 5.1.1. The pickle files also contain more information about the experiments (runtime, individual instance gaps, etc.). Please see `scrips` folder and its file "print_table_results_for_small_experiments.py" to obtain the results in the tables of the paper.
- family_setups_large_experiments_results contains the pickle files for the results of all the 108 instance types considered in Section 5.1.2 of the paper. The script "plotting_results_for_large_experiments.py" in `scripts` creates the figure in that section. The pickle files (as for the small experiments) contain more information that is used in the body of the paper (e.g., runtime, columns created, etc).

Furthermore, if deciding to run the code and create the figures and/or pickle files, new folders will be created with those files (to not delete the original ones provided in this package).