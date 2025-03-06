# Minitab-like Statistical Analysis Tool

A custom statistical analysis tool that provides functionality similar to Minitab, designed for Six Sigma and statistical analysis.

## Features

- **Basic Statistics**
  - Descriptive Statistics
  - Correlation Analysis
  - Probability Analysis

- **Advanced Statistics**
  - Hypothesis Testing
  - ANOVA
  - Regression Analysis

- **Six Sigma Tools**
  - Fishbone Diagram
  - FMEA Template
  - DPMO Calculator
  - Process Capability Analysis

- **Quality Control**
  - Control Charts (X-bar R, Individual, Moving Range)
  - Measurement System Analysis
  - Process Capability Studies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/imklchen/Minitab_like.git
cd Minitab_like
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python minitab_like_app_v6.py
```

## Dependencies

- Python 3.8+
- PyQt6
- pandas
- numpy
- scipy
- matplotlib
- statsmodels

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Create a Pull Request 