[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Submodular Dispatching with Multiple Vehicles

This project is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software in this repository are associated with the paper [Submodular Dispatching with Multiple Vehicles](https://doi.org/10.1287/ijoc.2024.0886) by Ignacio Erazo and Alejandro Toriello. 


## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2024.0886

https://doi.org/10.1287/ijoc.2024.0886.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{Erazo2025,
  author =        {Ignacio Erazo and Alejandro Toriello},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Submodular Dispatching with Multiple Vehicles}},
  year =          {2025},
  doi =           {10.1287/ijoc.2024.0886.cd},
  url =           {https://github.com/INFORMSJoC/2024.0886},
  note =          {Available for download at https://github.com/INFORMSJoC/2024.0886},
}  
```

## Description

This distribution includes the source code to reproduce the results of the paper "Submodular Dispatching with Multiple Vehicles" by Ignacio Erazo and Alejandro Toriello, published in the INFORMS Journal of Computing. The results of the paper are provided as .pickle files, and scripts to reproduce the tables and figures are also given.


## Requirements

In order to run the code, one needs to have all the python libraries installed (pip'ed). Please see/use the requirements.txt file provided in this package. 


## Results

The detailed results are provided in the `results` directory. This includes the figures for the paper in `paper_figures`, but also the raw results for the experiments in Section 5.1 of the article (Serial Machine Scheduling with Family Setups). These raw results (provided as pickle files) include more information than what is in the tables/figures of Section 5.1; and some of that information is referred to in the text (e.g., runtime). In `scripts` it is possible to replicate the figures of the paper, and the results in the tables; those figures/results will be added to new folders (such as to not eliminate the results and figures provided in this package). See the READMEs of each of the folders for more information.

## Data and Replication

- Since the paper evaluates the proposed methodology on two different experiments (family_setups and tactical design for Same-Day Delivery), the `src` directory contains a separate folder for each of them.
- The experiments and results can be replicated/obtained by running the code in `scripts`. Using "python file.py" or an IDE is enough.


## Support

The code is not supported.