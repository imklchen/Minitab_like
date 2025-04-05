# Minitab-like Application

A Python-based statistical analysis application inspired by Minitab, with features for descriptive statistics, quality control, and data visualization.

## Features

- **Basic Statistics**: Descriptive statistics, correlation analysis
- **Quality Control**: Control charts, capability analysis, probability analysis
- **Six Sigma Tools**: DMAIC tools, FMEA templates
- **Data Manipulation**: Load, save, and edit tabular data

## Requirements

- Python 3.6+
- Dependencies:
  - pandas
  - numpy
  - matplotlib
  - scipy
  - PyQt6 or PyQt5

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib scipy PyQt6
   ```
   or for PyQt5:
   ```
   pip install pandas numpy matplotlib scipy PyQt5
   ```

## Running the Application

### Basic Run

To run the basic application:

```
python run_direct.py
```

### Run with PyQt5

If you prefer to use PyQt5 instead of PyQt6:

```
python run_minitab_pyqt5.py
```

### Run with Sample Data

To start the application with sample data already loaded:

```
python run_app_with_sample.py
```

## Testing Probability Analysis

1. Run the application with sample data: `python run_app_with_sample.py`
2. Navigate to: Quality > Quality Tools > Probability Analysis
3. When prompted, select the "Height" column
4. The analysis will show:
   - A histogram with normal distribution curve
   - A Q-Q plot
   - Statistical results including mean, standard deviation, and normality test

## Project Structure

- `minitab_app/`: Main application package
  - `core/`: Core application components
  - `modules/`: Feature modules
    - `quality/`: Quality control features
    - `stats/`: Statistical analysis features
    - `ui/`: User interface components
- `sample_data/`: Sample datasets for testing
- `run_*.py`: Various scripts to run the application

## Current Status

The application is under active development. Currently implemented features:
- Basic data loading and manipulation
- Table view for data editing
- Descriptive statistics
- Control charts (X-bar R, Individual)
- Probability Analysis

## Troubleshooting

- **Event Loop Issues**: If you see "QCoreApplication::exec: The event loop is already running" messages, these are harmless warnings and can be ignored.
- **PyQt Version Compatibility**: The application supports both PyQt5 and PyQt6. If you encounter issues with one, try the other. 