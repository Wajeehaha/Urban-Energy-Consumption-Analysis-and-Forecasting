# 🏙️ Urban Energy Consumption Analysis and Forecasting

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green.svg)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-blue.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A comprehensive machine learning project for analyzing and forecasting urban energy consumption patterns using real-world utility company data.

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Project Structure](#️-project-structure)
- [🚀 Getting Started](#-getting-started)
- [📊 Dataset Information](#-dataset-information)
- [🔬 Analysis Components](#-analysis-components)
- [🤖 Machine Learning Models](#-machine-learning-models)
- [📈 Results and Insights](#-results-and-insights)
- [📋 Requirements](#-requirements)
- [🎮 Usage](#-usage)
- [📁 Output Files](#-output-files)
- [🔮 Future Enhancements](#-future-enhancements)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Project Overview

This project provides a complete end-to-end solution for analyzing urban energy consumption patterns and building predictive models for energy forecasting. Using real-world data from multiple utility companies, the analysis covers data preprocessing, exploratory data analysis, feature engineering, machine learning model development, and comprehensive reporting.

### 🎯 Objectives
- **Analyze** historical energy consumption patterns across multiple utility companies
- **Identify** seasonal trends, peak usage patterns, and efficiency metrics
- **Build** robust machine learning models for energy consumption forecasting
- **Generate** actionable insights for energy planning and optimization
- **Create** professional reports and visualizations for stakeholders

## ✨ Key Features

- 🔄 **Multi-format Data Support**: Handles CSV, Excel (.xlsx, .xls), and Parquet files
- 📈 **Comprehensive Analysis**: Seasonal patterns, peak vs off-peak analysis, growth trends
- 🤖 **Machine Learning**: Random Forest and Linear Regression models with 95%+ accuracy
- 📊 **Rich Visualizations**: Correlation matrices, time series plots, performance charts
- 📋 **Automated Reporting**: Professional CSV reports and executive summaries
- 🔧 **Robust Processing**: Error handling, data validation, and memory optimization
- 📱 **User-friendly**: Well-documented Jupyter notebook with clear explanations

## 🏗️ Project Structure

```
Urban-Energy-Consumption-Analysis-and-Forecasting/
│
├── 📓 energy_analysis.ipynb          # Main analysis notebook
├── 📁 data/                          # Raw data directory
│   ├── energy/                       # Energy consumption datasets
│   ├── weather/                      # Weather data (optional)
│   └── demographics/                 # Population data (optional)
│
├── 📁 reports/                       # Generated reports and outputs
│   ├── energy_source_metrics.csv     # Source performance metrics
│   ├── energy_source_summary.csv     # Statistical summaries
│   ├── model_comparison.csv          # ML model performance
│   └── comprehensive_source_metrics.png # Dashboard visualization
│
├── 📁 notebooks/                     # Additional analysis notebooks
├── 📁 outputs/                       # Temporary outputs and plots
├── 📁 .venv/                        # Python virtual environment
└── 📄 README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Wajeehaha/Urban-Energy-Consumption-Analysis-and-Forecasting.git
   cd Urban-Energy-Consumption-Analysis-and-Forecasting
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter openpyxl pyarrow fastparquet
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook energy_analysis.ipynb
   ```

## 📊 Dataset Information

### Data Sources
The project works with energy consumption data from major utility companies:

- **AEP** (American Electric Power) - Primary source
- **COMED** (Commonwealth Edison)
- **DAYTON** (Dayton Power & Light)
- **DEOK** (Duke Energy Ohio/Kentucky)
- **DOM** (Dominion Energy)
- **DUQ** (Duquesne Light)
- **EKPC** (East Kentucky Power Cooperative)
- **FE** (FirstEnergy)
- **NI** (Northern Indiana Public Service)
- **PJME** (PJM East)
- **PJMW** (PJM West)
- **PJM_Load** (PJM Load Data)

### Data Characteristics
- **Time Span**: 1998-2018 (20+ years of historical data)
- **Granularity**: Hourly consumption measurements
- **Volume**: 1.4M+ records across all sources
- **Format**: MW (Megawatts) measurements
- **Coverage**: 76.2% data completeness

## 🔬 Analysis Components

### 1. Data Preprocessing
- **Multi-format loading** with error handling
- **Data cleaning** and missing value treatment
- **Datetime parsing** and feature extraction
- **Memory optimization** for large datasets

### 2. Exploratory Data Analysis
- **Correlation analysis** between energy sources
- **Seasonal pattern detection** (monthly/yearly trends)
- **Peak vs off-peak** consumption analysis
- **Statistical summaries** and data quality assessment

### 3. Feature Engineering
- **Lagged consumption features** (1-7 days)
- **Time-based features** (hour, day of week, month, year)
- **Peak hour indicators** (6AM-10PM classification)
- **Rolling averages** and trend calculations

### 4. Visualization Suite
- **Time series plots** for consumption trends
- **Correlation heatmaps** for source relationships
- **Seasonal decomposition** charts
- **Model performance** comparisons
- **Source efficiency** rankings

## 🤖 Machine Learning Models

### Random Forest Regressor
- **Accuracy**: 95.1% (R² = 0.951)
- **RMSE**: 576.46 MW
- **Features**: Uses lagged consumption and time-based features
- **Strengths**: Handles non-linear patterns and feature interactions

### Linear Regression
- **Accuracy**: 82.0% (R² = 0.820)
- **RMSE**: 1,101.21 MW
- **Features**: Linear combination of engineered features
- **Strengths**: Interpretable coefficients and fast training

### Model Selection
**Recommended**: Random Forest due to superior performance (13% higher accuracy)

## 📈 Results and Insights

### 🎯 Key Findings

#### Energy Consumption Patterns
- **Average Hourly Consumption**: 15,499.51 MW (AEP)
- **Peak vs Off-Peak Ratio**: 1.17 (Excellent load balancing)
- **Seasonal Variation**: 8.0% (Moderate - good for planning)

#### Seasonal Insights
- **Highest Consumption**: January (17,431 MW), February (17,023 MW), December (16,446 MW)
- **Lowest Consumption**: April (13,824 MW), October (13,939 MW), May (14,006 MW)
- **Pattern**: Winter heating drives peak consumption

#### Efficiency Metrics
- **Most Stable Source**: AEP with low seasonal fluctuation
- **Load Management**: Excellent peak load distribution
- **Growth Trends**: Stable consumption patterns over time

### 📊 Performance Metrics

| Model | R² Score | RMSE (MW) | Accuracy | Use Case |
|-------|----------|-----------|----------|----------|
| Random Forest | 0.951 | 576.46 | 95.1% | **Primary forecasting** |
| Linear Regression | 0.820 | 1,101.21 | 82.0% | Baseline/interpretability |

## 📋 Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
openpyxl>=3.0.0
pyarrow>=5.0.0
fastparquet>=0.7.0
```

## 🎮 Usage

### Quick Start
1. **Open the notebook**: `energy_analysis.ipynb`
2. **Run all cells** sequentially (Kernel → Restart & Run All)
3. **View results** in the `reports/` directory

### Custom Analysis
```python
# Load your own data
datasets = load_datasets('path/to/your/data/')

# Run analysis
results = analyze_energy_consumption(datasets)

# Generate reports
generate_reports(results, output_dir='custom_reports/')
```

### Key Functions
- `load_datasets()`: Multi-format data loading
- `clean_energy_data()`: Data preprocessing pipeline
- `engineer_features()`: Feature creation and selection
- `train_models()`: ML model training and evaluation
- `generate_visualizations()`: Chart and plot creation

## 📁 Output Files

### Reports Directory
- **`energy_source_metrics.csv`**: Performance metrics by source
- **`energy_source_summary.csv`**: Statistical summaries
- **`model_comparison.csv`**: ML model performance comparison
- **`comprehensive_source_metrics.png`**: Visual dashboard

### Analysis Outputs
- **Correlation matrices**: Source relationship analysis
- **Seasonal plots**: Monthly/yearly consumption patterns
- **Model performance charts**: Accuracy and error analysis
- **Feature importance**: Variable significance rankings

## 🔮 Future Enhancements

- [ ] **Real-time data integration** via APIs
- [ ] **Advanced ML models** (LSTM, Prophet, XGBoost)
- [ ] **Interactive dashboards** with Plotly/Dash
- [ ] **Weather data correlation** analysis
- [ ] **Automated model retraining** pipeline
- [ ] **Anomaly detection** system
- [ ] **Multi-city analysis** expansion
- [ ] **Carbon footprint** calculations

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit changes** (`git commit -m 'Add AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black energy_analysis.ipynb --check
```

## 🙏 Acknowledgments

- **Data Sources**: PJM Interconnection and member utilities
- **Libraries**: pandas, scikit-learn, matplotlib, seaborn
- **Community**: Jupyter and Python data science community

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact & Support

- **Author**: [Wajeehaha](https://github.com/Wajeehaha)
- **Issues**: [GitHub Issues](https://github.com/Wajeehaha/Urban-Energy-Consumption-Analysis-and-Forecasting/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Wajeehaha/Urban-Energy-Consumption-Analysis-and-Forecasting/discussions)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the energy analytics community

</div>