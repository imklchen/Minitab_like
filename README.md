# Minitab-Like Statistical Analysis Tool

A comprehensive statistical analysis tool built with Python, providing functionality similar to Minitab. This application offers a wide range of statistical analyses, quality control tools, and design of experiments capabilities.

## Features

### Menu Structure

1. **File**
   - Open (supports CSV and Excel files)
   - Save (exports to CSV or Excel)
   - Exit

2. **Stat**
   - **Basic Statistics**
     - Descriptive Statistics
     - Correlation Analysis
     - Probability Analysis
   
   - **Advanced Statistics**
     - Hypothesis Testing
       - One-Sample t-Test
       - Two-Sample t-Test
       - Paired t-Test
     - ANOVA
       - One-Way ANOVA
       - Two-Way ANOVA
     - Regression Analysis
       - Simple Linear Regression
       - Multiple Linear Regression
     - Chi-Square Tests
       - Goodness of Fit
       - Test of Independence
       - Test of Homogeneity
   
   - **Design of Experiments (DOE)**
     - Create DOE
       - 2-level Factorial
       - Fractional Factorial
       - Response Surface
     - Analyze DOE

3. **Quality**
   - **Control Charts**
     - X-bar R Chart
     - Individual Chart
     - Moving Range Chart
   
   - **Capability Analysis**
     - Process Capability
     - Measurement System Analysis
       - Gage R&R Study
       - Linearity Study
       - Bias Study
       - Stability Study

4. **Six Sigma**
   - **DMAIC Tools**
     - Pareto Chart
     - Fishbone Diagram
     - FMEA Template
   
   - **Six Sigma Metrics**
     - DPMO Calculator
     - Sigma Level Calculator
     - Process Yield Analysis

5. **Calc**
   - **Random Data**
     - Normal
     - Binomial
     - Uniform
   
   - **Probability Distributions**
     - Poisson

## Requirements

- Python 3.x
- Required packages:
  - PyQt6
  - pandas
  - numpy
  - scipy
  - matplotlib
  - seaborn
  - statsmodels

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Minitab_like.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python minitab_like_app_v6.py
```

## Features Description

### Statistical Analysis
- Comprehensive descriptive statistics
- Various hypothesis tests
- ANOVA (One-way and Two-way)
- Regression analysis
- Chi-Square analysis

### Quality Control
- Various control charts
- Process capability analysis
- Measurement system analysis

### Design of Experiments
- 2-level factorial designs
- Fractional factorial designs
- Response surface designs

### Six Sigma Tools
- DMAIC tools implementation
- Process performance metrics
- Quality improvement templates

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with PyQt6 for the graphical interface
- Statistical computations powered by SciPy and NumPy
- Visualization capabilities provided by Matplotlib and Seaborn 