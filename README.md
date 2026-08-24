# 🌬️ Energy Drought Projections & Wind-Solar Hybrid Complementarity in Indonesia

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![CDO](https://img.shields.io/badge/CDO-Climate%20Data%20Operators-green)
![RegCM](https://img.shields.io/badge/Model-RegCM5_MOLOCH-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview
This repository contains the data processing pipeline, statistical bias-correction, and spatial analysis codebase for projecting **Energy Droughts (Dunkelflaute)** and evaluating wind-solar hybrid system complementarity across the Indonesian Maritime Continent. 

By executing high-resolution dynamical downscaling, this project translates complex global climate variables into actionable renewable energy metrics. The analytical framework successfully delineates Indonesia into distinct hybrid complementarity zones using machine learning (K-Means), proving that coastal micro-grids are highly resilient, while inland onshore wind installations face severe vulnerability to weather anomalies.

**Data Engineering Scale:**
- **Raw Data Handled:** Over 40+ Terabytes of CMIP6 global inputs.
- **Simulated Period:** 168 years total (1991-2014 historical baseline + 4 future scenarios x 36 years to 2050).
- **Spatiotemporal Resolution:** 6-hourly; 25-km grid resolution.

## 🎯 Analytical Framework & Methodology
1. **Dynamical Downscaling (RegCM5):** Downscaling CMIP6 Global Climate Models (EC-Earth3 and NorESM2-MM) utilizing the non-hydrostatic MOLOCH core in RegCM5 to capture complex local circulations over the Indonesian archipelago.
2. **Statistical Bias Correction:** Implementing Quantile Mapping (QM) for historical data and Quantile Delta Mapping (QDM) for future climate scenarios (SSP1-2.6, SSP2-4.5, SSP3-7.0, and SSP5-8.5) against ERA5 reanalysis data.
3. **Capacity Factor (CF) Conversion:** Converting 100m wind speeds and surface solar radiation (rsds) into daily CF based on utility-scale Siemens SWT-3.6-130 turbine and monocrystalline PV specifications.
4. **Energy Drought Profiling:** Identifying compound energy droughts using Energy Drought Frequency (EDF) and maximum consecutive Energy Drought Duration (EDD) indices.
5. **Complementarity & Clustering:** Evaluating system synergy using Spearman's Rank Correlation Coefficient (SRCC) and Composite Variability Index (CVI), followed by K-Means clustering (k=2) to delineate optimal spatial hybrid zones.

## 🗂️ Repository Structure
To maintain strict code reproducibility, the analytical scripts are modularized:

    ├── data/
    │   ├── raw/                  # CMIP6 and ERA5 reference data (Not uploaded due to size)
    │   ├── processed/            # Bias-corrected and compressed CF datasets
    │   └── outputs/              # Generated spatial maps and statistical plots
    ├── notebooks/
    │   ├── 01_gcm_evaluation.ipynb         # Taylor Diagrams and TSS for model selection
    │   ├── 02_bias_correction.ipynb        # QM and QDM implementation via Python/CDO
    │   ├── 03_energy_conversion.ipynb      # Wind/Solar CF thermodynamic conversion
    │   ├── 04_energy_drought_analysis.ipynb# EDF and EDD thresholding
    │   └── 05_hybrid_complementarity.ipynb # SRCC, CVI, and K-Means Clustering
    ├── scripts/
    │   ├── cdo_pipeline.sh       # Bash scripts for automated CDO pre-processing
    │   └── utils.py              # Helper functions for geospatial operations (Xarray)
    ├── requirements.txt          # Conda/Pip environment dependencies
    └── README.md

## 🚀 Reproducibility & Installation
To reproduce this environment and run the geospatial analysis locally:

1. **Clone the repository:**
    git clone [https://github.com/naufaljouhary/energy-drought-complementarity.git](https://github.com/naufaljouhary/energy-drought-complementarity.git)
    cd energy-drought-complementarity

2. **Create the virtual environment:**
    conda create --name climate-env python=3.9
    conda activate climate-env
    pip install -r requirements.txt
    *(Ensure Climate Data Operators (CDO) is installed via your system's package manager).*

3. **Running the Pipeline:**
   Launch JupyterLab and execute the modularized notebooks sequentially in the `/notebooks` directory.

## 📬 Contact
**Naufal A. Jouhary** - Climate Data Analyst
- LinkedIn: [naufal-jouhary](https://www.linkedin.com/in/naufal-jouhary-71900622b)
- Email: naufaljouhary@gmail.com
