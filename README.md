# 🌪️ Dunkelflaute Risk Mitigation & Wind-Solar Complementarity in Indonesia

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: RegCM5](https://img.shields.io/badge/Downscaling-RegCM5_MOLOCH-red.svg)](https://github.com/ICTP/RegCM)

## 📖 Project Overview
This repository contains the core analytical scripts and spatial models for assessing the risk of **Dunkelflaute** (compound energy droughts involving low wind and solar generation) across the Indonesian Maritime Continent. 

While wind and solar spatial complementarity can optimize the renewable energy mix, this research utilizes high-resolution dynamical downscaling to mathematically demonstrate the critical vulnerability of extreme weather anomalies. The ultimate techno-economic and policy implication derived from this modeling strongly suggests the absolute necessity of integrating **Firm Capacity (e.g., Nuclear Power)** to ensure energy sovereignty and grid reliability.

**Author:** Naufal Jouhary (Applied Climatology, IPB University)

## 🎯 Key Objectives & Methodology
1. **CMIP6 Model Evaluation:** Validating 7 Global Climate Models against ERA5 reanalysis data using Taylor Diagrams and Taylor Skill Scores (TSS) to select the optimal boundary conditions.
2. **Dynamical Downscaling (RegCM5):** Utilizing the non-hydrostatic MOLOCH core in RegCM5 to capture complex orographic effects and convection-permitting local winds at a 5 km high-resolution nest.
3. **Statistical Bias Correction:** Implementing Empirical Quantile Mapping (EQM) for historical data and Quantile Delta Mapping (QDM) for future climate projections (SSP scenarios).
4. **Dunkelflaute & Complementarity Profiling:** Applying the Duration Given Intensity (DGI) framework to profile energy droughts, and evaluating synergy using Spearman's Rank Correlation Coefficient (SRCC) and Composite Variability Index (CVI).
5. **Policy Implication:** Comparing the Levelized Cost of Storage (LCOS) for long-duration batteries versus Nuclear firm capacity during extreme Dunkelflaute events.

## 🗄️ Repository Structure
To maintain reproducibility, scripts are modularized into specific tasks:

```text
├── data/                  # Dummy/Sample NetCDF data for testing
├── outputs/               # Generated high-res spatial maps, Taylor Diagrams, and PDF curves
├── scripts/
│   ├── 01_plot_topography.py       # DEM data extraction, subsetting, and spatial hillshade plotting
│   ├── 02_model_evaluation.py      # RMSE, Correlation, and Taylor Diagram/TSS generation
│   ├── 03_bias_correction.py       # EQM and QDM algorithms for simulated climate data
│   ├── 04_capacity_factor.py       # Wind/Solar CF conversion based on Siemens SWT-3.6-130 turbine curve
│   └── 05_dunkelflaute_cvi.py      # Energy drought thresholding and SRCC/CVI spatial analysis
├── requirements.txt       # Python environment dependencies
└── README.md              # Project documentation
```

## 📊 Data Provenance & Management
* **Global Climate Models (GCM):** Data sourced from CMIP6 scenarios (SSP1-2.6, SSP2-4.5, SSP3-7.0, and  SSP5-8.5).
* **Reference Data:** ERA5 Reanalysis from ECMWF.
* **Topography:** GMTED subsetted to the Indonesian maritime domain (Lat: -12 to 10, Lon: 94 to 142) to optimize computational efficiency during RegCM5 nesting.

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone [https://github.com/naufaljouhary/dunkelflaute-idn-regcm.git](https://github.com/naufaljouhary/dunkelflaute-idn-regcm.git)
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the modules sequentially from the `scripts/` directory.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
