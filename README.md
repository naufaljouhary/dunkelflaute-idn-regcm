# 🌪️ Dunkelflaute Risk Mitigation & Wind-Solar Complementarity in Indonesia

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: RegCM4](https://img.shields.io/badge/Data-ICTP_RegCM4-green.svg)](http://clima-dods.ictp.it/regcm4/)

## 📖 Project Overview
This repository contains the core analytical scripts and spatial models for assessing the risk of **Dunkelflaute** (periods of low wind and low solar energy generation) across the Indonesian archipelago. 

While wind and solar complementarity can optimize the renewable energy mix, this research mathematically demonstrates the critical vulnerability of extreme weather anomalies. The ultimate policy implication derived from this modeling strongly suggests the absolute necessity of integrating **Firm Capacity (e.g., Nuclear Power)** into the grid to ensure energy sovereignty and grid stability.

**Author:** Naufal Jouhary (Applied Climatology, IPB University)

## 🎯 Key Objectives
1. **Spatial Complementarity:** Mapping the spatio-temporal dynamics of wind speed (`sfcWind`) and solar radiation (`rsds`) across tropical regions.
2. **Model Evaluation:** Validating Global/Regional Climate Models (e.g., EC-Earth3, CanESM5) against ERA5 reanalysis data using Taylor Diagrams and Taylor Skill Scores (TSS).
3. **Dunkelflaute Profiling:** Identifying extreme weather anomalies that cripple variable renewable energy (VRE) generation.
4. **Policy Implication:** Providing empirical data modeling to justify the inclusion of stable firm capacity in Indonesia's energy transition roadmap.

## 🗄️ Repository Structure
To maintain reproducibility, scripts are modularized into specific tasks:

```text
├── data/                  # Dummy/Sample NetCDF data for testing (Raw data excluded due to size)
├── outputs/               # Generated high-res spatial maps and Taylor Diagrams
├── scripts/
│   ├── 01_plot_topography.py       # DEM data extraction, subsetting, and spatial hillshade plotting
│   ├── 02_model_evaluation.py      # RMSE, Correlation, and Taylor Diagram generation
│   └── 03_dunkelflaute_analysis.py # VRE thresholding and extreme anomaly detection
├── requirements.txt       # Python environment dependencies
└── README.md              # Project documentation
