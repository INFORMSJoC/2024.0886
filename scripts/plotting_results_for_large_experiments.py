# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
import pickle
from scipy.stats import gmean
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


def open_pickle_object_and_pass_data(
        file_path: str
        ) -> object:
    """
    Function to open the pickle files and pass the content as a python object.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return

OBTAIN_PAPER_FIGURES = True

if __name__ == '__main__':
    # start of path for the pickle files that have the results from the
    # large family setups experiments
    prefix_path = "results/family_setups_large_experiments_results/"
    
    # process the data for 80 orders; for the 12 scenarios and m = 5, 10, 15 machines
    run_time_colgen_n80 = []
    array_colgen1_n80 = []
    array_interval_n80 = []
    best_heur_n80 = []
    for machines in [5, 10, 15]:
        for Q in [5, 10, 15]:
            for Uf in [51, 101]:
                for Ur in [10, 26]:
                    path_to_read = prefix_path + \
                        f'results_(80, {machines}, {Q}, (1, 101), (0, {Uf}), (0, {Ur}))' + \
                        "_120_sec.pickle"
                    data = open_pickle_object_and_pass_data(file_path=path_to_read)
                    array_with_gaps_for_methods = data[3]

                    # colgen heuristic geometric mean gaps as a percent
                    array_colgen1_n80.append(100*gmean(array_with_gaps_for_methods[1]))

                    # colgen heuristic geometric mean gaps as a percent
                    array_interval_n80.append(100*gmean(array_with_gaps_for_methods[3]))

                    # colgen heuristic geometric mean gaps as a percent
                    best_heur_n80.append(100*gmean(array_with_gaps_for_methods[4]))

                    # runtime of the column generation LP solve
                    run_time_colgen_n80.append(np.mean(data[4][0]))


    # process the data for 120 orders; for the 12 scenarios and m = 5, 10, 15 machines
    run_time_colgen_n120 = []
    array_colgen1_n120 = []
    array_interval_n120 = []
    best_heur_n120 = []
    for machines in [5, 10, 15]:
        for Q in [5, 10, 15]:
            for Uf in [51, 101]:
                for Ur in [10, 26]:
                    path_to_read = prefix_path + \
                        f'results_(120, {machines}, {Q}, (1, 101), (0, {Uf}), (0, {Ur}))' + \
                        "_120_sec.pickle"
                    data = open_pickle_object_and_pass_data(file_path=path_to_read)
                    array_with_gaps_for_methods = data[3]

                    # colgen heuristic geometric mean gaps as a percent
                    array_colgen1_n120.append(100*gmean(array_with_gaps_for_methods[1]))

                    # colgen heuristic geometric mean gaps as a percent
                    array_interval_n120.append(100*gmean(array_with_gaps_for_methods[3]))

                    # colgen heuristic geometric mean gaps as a percent
                    best_heur_n120.append(100*gmean(array_with_gaps_for_methods[4]))

                    # runtime of the column generation LP solve
                    run_time_colgen_n120.append(np.mean(data[4][0]))


    # process the data for 160 orders; for the 12 scenarios and m = 5, 10, 15 machines
    run_time_colgen_n160 = []
    array_colgen1_n160 = []
    array_interval_n160 = []
    best_heur_n160 = []
    for machines in [5, 10, 15]:
        for Q in [5, 10, 15]:
            for Uf in [51, 101]:
                for Ur in [10, 26]:
                    path_to_read = prefix_path + \
                        f'results_(160, {machines}, {Q}, (1, 101), (0, {Uf}), (0, {Ur}))' + \
                        "_120_sec.pickle"
                    data = open_pickle_object_and_pass_data(file_path=path_to_read)
                    array_with_gaps_for_methods = data[3]

                    # colgen heuristic geometric mean gaps as a percent
                    array_colgen1_n160.append(100*gmean(array_with_gaps_for_methods[1]))

                    # colgen heuristic geometric mean gaps as a percent
                    array_interval_n160.append(100*gmean(array_with_gaps_for_methods[3]))

                    # colgen heuristic geometric mean gaps as a percent
                    best_heur_n160.append(100*gmean(array_with_gaps_for_methods[4]))

                    # runtime of the column generation LP solve
                    run_time_colgen_n160.append(np.mean(data[4][0]))
    
    print(f'Average and max of running time (in seconds) n=80: {np.mean(run_time_colgen_n80):.2f} and {np.max(run_time_colgen_n80):.2f} ')
    print(f'Average and max of running time (in seconds) n=120: {np.mean(run_time_colgen_n120):.2f} and {np.max(run_time_colgen_n120):.2f} ')
    print(f'Average and max of running time (in seconds) n=160: {np.mean(run_time_colgen_n160):.2f} and {np.max(run_time_colgen_n160):.2f} ')
    
    if OBTAIN_PAPER_FIGURES:
        PATH_TO_STORE_FIGURES = 'results/figures_family_setups_large_obtained_by_script/'
        os.makedirs(name=PATH_TO_STORE_FIGURES,
                    exist_ok=True)
        # Data divided
        colors = ['blue', 'red', 'green']
        markers = ['+', 'o', '^']
        labels = ['n = 80', 'n = 120', 'n = 160']
        nam = 'family_setups__experiments_large_'
        plot_1_data = [array_colgen1_n80[:12], array_colgen1_n120[:12], array_colgen1_n160[:12]]
        plot_3_data = [array_colgen1_n80[12:24], array_colgen1_n120[12:24], array_colgen1_n160[12:24]]
        plot_5_data = [array_colgen1_n80[24:], array_colgen1_n120[24:], array_colgen1_n160[24:]]
        
        plot_2_data = [array_interval_n80[:12], array_interval_n120[:12], array_interval_n160[:12]]
        plot_4_data = [array_interval_n80[12:24], array_interval_n120[12:24], array_interval_n160[12:24]]
        plot_6_data = [array_interval_n80[24:], array_interval_n120[24:], array_interval_n160[24:]]
        
        postfix_for_image = ['a', 'b', 'c', 'd', 'e', 'f']
        for index, data in enumerate([plot_1_data, plot_2_data, plot_3_data, 
                    plot_4_data, plot_5_data, plot_6_data]):
            ax = plt.subplot()
            ax.set_ylim(100, 180)
            ax.tick_params(axis='both', which='major', labelsize=14)
            for i, entry in enumerate(data):
                ax.plot([j for j in range(1, 13)], entry,
                    marker=markers[i], color=colors[i], linestyle='dashed',
                    label=labels[i])
            if index == 0:
                ax.legend(prop={'size': 14})
            plt.savefig(PATH_TO_STORE_FIGURES + nam + f'1{postfix_for_image[index]}' + '.pdf')
            plt.show()
        