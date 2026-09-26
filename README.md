# Open Metrics and Visual Diagnostics for Solar Wind Model Validation

This repository provides metrics and visual diagnostics for assessing the capabilities of solar wind models. It contains widely used procedures for comparing solar wind model outputs against near-Earth in situ measurements. The repository complements the [CAMEL Ambient Solar Wind Validation Dashboard](https://ccmc.gsfc.nasa.gov/camel/AmbientSolarWind/home), developed and maintained at the [Community Coordinated Modeling Center (CCMC)](https://ccmc.gsfc.nasa.gov), by enabling users to reproduce assessment results and modify procedures. 

The validation procedures include point-to-point metrics, threshold-based event validation, and peak-detection-based event validation. This effort is part of ongoing community work through the [ISWAT H1-01 Team](https://iswat-cospar.org/h1-01) to support unified capability assessment for ambient solar wind models across the heliophysics and space weather community.

## Overview

The repository is organized around a set of Python modules and a Jupyter notebook workflow. The modules include data preparation, metrics, and visual diagnostics, while the notebooks provide end-to-end examples of how to use the modules on solar wind model outputs included in the CAMEL Dashboard.

Core functionality includes:
- preprocessing and alignment of modeled and observed solar wind time series
- point-to-point metrics
- threshold-based event validation with user-selected thresholds
- peak-detection-based event validation focused on timing and amplitude errors
- multi-model comparison against a reference baseline model
- visual diagnostics and PDF figure export

## Python Modules

The main code is organized into three Python modules:

- `data_processing.py` contains functions for cleaning, preprocessing, and interpolating modeled and observed solar wind time series.
- `skill_scores.py` includes the metrics used in the CAMEL Dashboard and throughout the notebooks.
- `visualizations.py` contains the Plotly routines used to reproduce the CAMEL Dashboard visualizations.

## Jupyter Notebooks

The notebook workflow provides a step-by-step guide for reproducing the assessment results in the CAMEL Ambient Solar Wind Dashboard:

1. `01_api_and_preprocessing.ipynb` uses an API to retrieve modeled and observed solar wind time series and saves the example datasets.
2. `02_point_to_point_metrics.ipynb` computes point-to-point metrics and plots visual diagnostics.
3. `03_binary_event_metrics.ipynb` computes threshold-based event metrics and plots visual diagnostics.
4. `04_peak_analysis.ipynb` computes peak-detection-based event metrics and plots visual diagnostics.
5. `05_multi_model_comparison.ipynb` compares multiple model outputs side by side against in situ measurements.

The notebooks save generated PDF figures under `results/output_pdfs/`.

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install dependencies

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes `kaleido`, which is required for Plotly figure export.

### 3. Run the notebooks

Start Jupyter and open the notebook series:

```bash
jupyter notebook notebooks/01_api_and_preprocessing.ipynb
```

Start with `01_api_and_preprocessing.ipynb` to create the processed example data before computing metrics and generating visualizations in the later notebooks.

## Tests

Run the peak-matching regression tests after installing NumPy, pandas, and SciPy:

```bash
python -m unittest discover -s tests -v
```

## Citation

A publication describing the current implementation is in preparation. Until that publication is available, please cite the following prior work:

Reiss, M. A., et al. (2022), *Unifying the validation of ambient solar wind models*, **Advances in Space Research**. [https://doi.org/10.1016/j.asr.2022.05.026](https://doi.org/10.1016/j.asr.2022.05.026)

## Acknowledgements

This effort was developed in collaboration with the [ISWAT H1-01 Team](https://iswat-cospar.org/h1-01) and builds on the CAMEL validation framework developed at the [CCMC](https://ccmc.gsfc.nasa.gov/).

We are grateful for the feedback and discussions gathered through COSPAR ISWAT meetings.

## Contributors

- Martin A. Reiss
- Maya Levisohn
- Rick Mullinix

## License

This project is released under the NASA Open Source Agreement, Version 1.3 (NOSA 1.3). See `LICENSE` for the full license text.

## Contact

Martin A. Reiss ([email](mailto:martin.a.reiss@outlook.com))
