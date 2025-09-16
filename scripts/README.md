## Organization

This folder is divided into 4 files, one of these files is related to the Same-Day Delivery experiments:
- run_tactical_design_sdd_experiments.py = this file runs the experiments in Section 5.2; and also can output the figures used in the paper (see flags in the file).

The other 3 files correspond to the experiments in Section 5.1:
- print_table_results_for_small_experiments.py = this file looks at the .pickle files carrying on the granular results for the results of Section 5.1.1, and outputs as text the results for the (ordered) rows of the tables in that Subsection.
- plotting_results_for_large_experiments.py = this file looks at the .pickle files carrying on the granular results for the results of Section 5.1.2, and outputs the figures detailing the results in that subsection.
- run_family_setups_experiments.py = this file runs the experiments in Subsections 5.1.1 and 5.1.2; and stores the results into pickle files for further processing with the other script files.