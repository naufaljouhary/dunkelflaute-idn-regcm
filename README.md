# 🌬️ Energy Drought Projections & Wind-Solar Hybrid Complementarity in Indonesia

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![CDO](https://img.shields.io/badge/CDO-Climate%20Data%20Operators-green)
![RegCM](https://img.shields.io/badge/Model-RegCM5_MOLOCH-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview
This repository contains the data processing pipeline, statistical bias-correction, and spatial analysis codebase for projecting **Energy Droughts (Dunkelflaute)** and evaluating wind-solar hybrid system complementarity across the Indonesian Maritime Continent. 

By executing high-resolution dynamical downscaling, this project translates complex global climate variables into actionable renewable energy metrics. The analytical framework delineates Indonesia into distinct hybrid complementarity zones using machine learning (K-Means), proving that coastal micro-grids remain resilient, while inland onshore wind installations face significant vulnerability to weather anomalies.

**Data Engineering Scale:**
- **Raw Data Handled:** Over 40+ Terabytes of CMIP6 global climate inputs.
- **Simulated Period:** 168 years total (1991–2014 historical baseline + 4 future SSP scenarios × 36 years through 2050).
- **Spatiotemporal Resolution:** 6-hourly temporal resolution; 25-km spatial grid.

## 🎯 Analytical Framework & Methodology
1. **Dynamical Downscaling (RegCM5):** Downscaling CMIP6 Global Climate Models (EC-Earth3 and NorESM2-MM) utilizing the non-hydrostatic MOLOCH dynamical core in RegCM5.
2. **Statistical Bias Correction:** Implementing Quantile Mapping (QM) for historical data and Quantile Delta Mapping (QDM) with a 91-day sliding window for future climate scenarios (SSP1-2.6, SSP2-4.5, SSP3-7.0, and SSP5-8.5) against ERA5 reanalysis data.
3. **Capacity Factor (CF) Conversion:** Converting 100m wind speeds and surface solar radiation (rsds) into daily CF using Siemens SWT-3.6-130 turbine power curves and cell temperature thermodynamic models.
4. **Energy Drought Profiling:** Identifying compound energy droughts using Energy Drought Frequency (EDF) and maximum consecutive Energy Drought Duration (EDD) with a 1-day pooling algorithm.
5. **Complementarity & Clustering:** Evaluating spatial synergy using Spearman's Rank Correlation Coefficient (SRCC) and Composite Variability Index (CVI), followed by K-Means clustering (k=2) to delineate optimal regional hybrid zones.

## 🗂️ Repository Structure
To maintain strict reproducibility, the pipeline is modularized as follows:

```text
├── data/
│   ├── raw/                                # CMIP6 and ERA5 reference data (Not tracked via Git)
│   ├── processed/                          # Bias-corrected intermediate datasets (Not tracked via Git)
│   └── outputs/                            # Generated spatial maps, figures, and Bappenas summary tables
│       └── tables/                         # Regional tabular extractions (.xlsx)
├── notebooks/
│   ├── 01_gcm_evaluation.ipynb             # Taylor Diagrams and TSS spatial validation
│   ├── 02_bias_correction.ipynb            # Spatial absolute/bias matrices and CDF validation
│   ├── 03_energy_conversion.ipynb          # Climatological & seasonal Capacity Factor maps and trends
│   ├── 04_energy_drought_analysis.ipynb    # EDF and EDD thresholding with unviable masking
│   └── 05_hybrid_complementarity.ipynb     # SRCC, CVI, and K-Means regional classification
├── scripts/
│   ├── Template_EC-Earth3_RegCM5.in        # Master Namelist for EC-Earth3 dynamical downscaling
│   ├── Template_NorESM2-MM_RegCM5.in       # Master Namelist for NorESM2-MM dynamical downscaling
│   ├── 01_bias_correction_qm_qdm.py        # QDM engine with 91-day sliding window and multi-windowing
│   ├── 02_energy_conversion_aggregation.py # Thermodynamic CF conversion & daily mean aggregation
│   ├── 03_energy_drought_extraction.py     # EDF & EDD pooling calculations (Qu et al.)
│   ├── 04_hybrid_complementarity_engine.py # SRCC and CVI computation engine
│   └── 05_continuous_trend_analysis.py     # 60-year continuous Theil-Sen trend engine
├── requirements.txt                        # Python environment dependencies
├── LICENSE                                 # MIT License
└── README.md
```

## 🚀 Reproducibility & Installation
To set up this environment and execute the pipeline:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/naufaljouhary/energy-drought-complementarity.git](https://github.com/naufaljouhary/energy-drought-complementarity.git)
   cd energy-drought-complementarity
   ```

2. **Create the virtual environment:**
   ```bash
   conda create --name climate-env python=3.9
   conda activate climate-env
   pip install -r requirements.txt
   ```

3. **Running the Pipeline:**
   - Execute the heavy processing tasks on HPC using scripts in `/scripts`.
   - Launch JupyterLab to reproduce all figures and maps using the notebooks in `/notebooks`.

## 📬 Contact
**Naufal A. Jouhary** - Climate Data Analyst
- LinkedIn: [naufal-jouhary](https://www.linkedin.com/in/naufal-jouhary-71900622b)
- Email: naufaljouhary@gmail.com
