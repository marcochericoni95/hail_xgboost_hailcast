# European hail days from XGBoost and HAILCAST  
### Figure-generation code for Nature Climate Change

This repository contains the Python scripts used to generate all figures
presented in the manuscript submitted to Nature Climate Change:
"Divergent European hail projections from machine-learning and physically based models under global warming"
which analyses European hail occurrence derived from a physically based HAILCAST model coupled to convection-permitting climate simulations and
a machine-learning model (XGBoost) applied to the same simulations.

The repository includes **only figure-generation code**.
Hail output from both XGBoost and HAILCAST are archived separately on Zenodo.

This repository **does not** contain:
- raw meteorological data
- model training code

---

## Data availability

Hail outputs from XGBoost and HAILCAST are publicly available via Zenodo at:

https://doi.org/10.5281/zenodo.18243342

All scripts in this repository assume that the Zenodo dataset has been downloaded locally.

---

## Repository contents

The `scripts/` directory contains one script per figure:

- `fig01.py` — generates **Figure 1** of the manuscript  
- `fig02.py` — generates **Figure 2** of the manuscript  
- `fig03.py` — generates **Figure 3** of the manuscript  

Each script is self-contained and generates the corresponding figure.

---

## Software environment

A reference software environment is provided in `environment.yml`.

To create the environment:

```bash
conda env create -f environment.yml
conda activate hail-xgboost

