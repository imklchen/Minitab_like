# Minitab-Like Application Test Guide

## Getting Started
1. Run `dist/MinitabLikeApp.exe`
2. Use the sample data files provided:
   - control_chart_data.csv (Control Charts)
   - chi_square_data.csv (Chi-Square Tests)
   - sample_data.csv (General Analysis)
   - See doe_test_cases.md for DOE examples

## Test Data and Functions

### 1. Basic Statistics
#### Descriptive Statistics
```
Sample Data (from sample_data.csv):
Data,Category,Value,Group
10.2,A,Product1,Control
10.5,A,Product2,Control
10.1,B,Product1,Treatment
10.3,B,Product2,Treatment
10.4,A,Product1,Control
10.3,A,Product2,Control
10.6,B,Product1,Treatment
10.2,B,Product2,Treatment
10.4,A,Product1,Control
10.5,A,Product2,Treatment
```
- Go to: Stat > Basic Statistics > Descriptive Statistics
- Load sample_data.csv
- All columns (Data, Category, Value, Group) should appear in the grid window
- Select 'Data' column for descriptive statistics
- Expected Results:
  - Count = 10
  - Mean = 10.350000
  - Standard Deviation = 0.158114
  - Minimum = 10.100000
  - Q1 (25%) = 10.225000
  - Median (50%) = 10.350000
  - Q3 (75%) = 10.475000
  - Maximum = 10.600000
  - Check histogram for normal distribution

### 2. Process Yield Analysis
#### Test Case 1
```
Sample Data:
LSL: 10
USL: 50
Mean: 30
StdDev: 5
Number of Units: 100
Opportunities per Unit: 1
```
- Go to: Six Sigma > Six Sigma Metrics > Process Yield Analysis
- Enter the values above
- Expected Results:
  - Process Yield ≈ 99.99%
  - DPMO ≈ 63.34
  - Sigma Level ≈ 5.33
  - Capability Indices:
    - Cp ≈ 1.333
    - Cpu ≈ 1.333
    - Cpl ≈ 1.333
    - Cpk ≈ 1.333

#### Test Case 2
```
Sample Data:
LSL: 95
USL: 105
Mean: 100
StdDev: 2
Number of Units: 100
Opportunities per Unit: 1
```
- Go to: Six Sigma > Six Sigma Metrics > Process Yield Analysis
- Enter the values above
- Expected Results:
  - Process Yield ≈ 98.76%
  - DPMO ≈ 12419.33
  - Sigma Level ≈ 3.74
  - Capability Indices:
    - Cp ≈ 0.833
    - Cpu ≈ 0.833
    - Cpl ≈ 0.833
    - Cpk ≈ 0.833

### 3. Chi-Square Tests
There are three types of Chi-Square tests available, each requiring its own specific data file format. Follow these detailed steps for each test:

Important Pre-Test Steps:
- Ensure all previous results windows are closed
- Data should be properly formatted in CSV files
- Column names should match exactly as shown in sample data
- No missing values are allowed in the data

Application Interface Layout:
- Main Window: Dark theme interface with menu bar at top
- Data View: Grid display showing loaded data in upper portion
- Results Area: Session window in lower portion showing:
  * Test results in text format
  * Statistical details in tabular format
  * Visualizations in separate windows

Navigation: Stat > Advanced Statistics > Chi-Square Tests
Test Selection Dialog:
- A dialog will appear with three options:
  * Goodness of Fit
  * Test of Independence
  * Test of Homogeneity
- Select the desired test type

#### 1. Goodness of Fit Test
Test the following data using Chi-Square Goodness of Fit test:

```
Category,Observed
1,15
2,17
3,16
4,19
5,15
6,18
```

* The session window should show exactly:

```
Chi-Square Goodness of Fit Test Results

Test Information:
Number of categories: 6
Total observations: 100

Test Statistics:
Chi-square statistic: 0.8000
Degrees of freedom: 5
p-value: 0.9770

Category Details:
Category Observed Expected Contribution
---------------------------------------------
1        15      16.67    0.1667
2        17      16.67    0.0067
3        16      16.67    0.0267
4        19      16.67    0.3267
5        15      16.67    0.1667
6        18      16.67    0.1067

Decision:
Fail to reject the null hypothesis at α = 0.05

Note: The null hypothesis is that the observed frequencies follow the specified probabilities.

* A bar chart will appear showing:
- Title: "Chi-Square Goodness of Fit Test Observed vs Expected Frequencies"
- Blue bars: Observed Frequencies
- Red bars: Expected Frequencies
- X-axis: Categories (1-6)
- Y-axis: Frequency values (0-20)
- Legend showing "Observed" and "Expected"
```

Specific Troubleshooting for Goodness of Fit Test:
1. Data Format Issues:
   - Ensure Category column contains only numbers 1-6
   - Ensure Observed column contains only positive integers
   - Verify total observations sum to 100

2. Common Problems:
   - Incorrect expected frequency calculations
   - Rounding errors in chi-square statistic
   - Wrong degrees of freedom

3. Visualization Issues:
   - If bar chart doesn't appear, close and reopen test
   - Check if any other windows are blocking the chart
   - Use window controls to resize if needed

Visualization Metrics:
- Y-axis range: 0 to 20
- Bar width: 0.35
- Grid alpha: 0.3
- Bar alpha: 0.7
- DPI: 100

Power Analysis:
- Power (at α = 0.05) = 0.0880
- Required sample size for 0.80 power = 1625

#### 2. Test of Independence
Data File: independence_test_data.csv
```
Gender,Product,Count
Male,Product A,30
Male,Product B,25
Male,Product C,20
Female,Product A,35
Female,Product B,30
Female,Product C,25
```

Testing Steps:
1. Launch MinitabLikeApp.exe
2. Go to: File > Open (or use Ctrl+O)
3. Select and load independence_test_data.csv
4. Navigate to: Stat > Advanced Statistics > Chi-Square Tests (or Alt+S, A, C)
5. Select "Test of Independence" from the test type dialog
6. When prompted for variable selection:
   - First prompt: Select "Gender" as the first variable
   - Second prompt: Select "Product" as the second variable

Expected Results:
The session window should show exactly:
```
Chi-Square Test of Independence Results

Test Information:
Variables: Gender and Product
Number of rows: 2
Number of columns: 3

Test Statistics:
Chi-square statistic: 0.0313
Degrees of freedom: 2
p-value: 0.9845

Contingency Table:
Product      Product A  Product B  Product C  Row Total
Gender
Female            35         30         25         90
Male              30         25         20         75


Expected Frequencies:
Product      Product A  Product B  Product C  Row Total
Gender
Female       35.45      30.00      24.55      90.00
Male         29.55      25.00      20.45      75.00


Chi-Square Contributions:
Product      Product A  Product B  Product C
Gender
Female       0.0058     0.0000     0.0084
Male         0.0070     0.0000     0.0101


Decision:
Fail to reject the null hypothesis at α = 0.05

Note: The null hypothesis is that the variables are independent.
```

Visualizations:
- Two heat maps will appear showing:
  * Left plot: "Observed Frequencies"
    - Shows actual counts in each cell
    - Values range from 20 to 35
  * Right plot: "Expected Frequencies"
    - Shows calculated expected values
    - Values range from 20.45 to 35.45

Additional Statistical Metrics:
```
Effect Size Measures:
- Cramer's V: 0.0387
- Contingency Coefficient: 0.0547
- Phi Coefficient: 0.0547

Standardized Residuals:
Product     Product A  Product B  Product C
Gender                                    
Male        -0.2062   -0.0905   -0.0922
Female       0.1978    0.0870    0.0885

Power Analysis:
- Observed Power (at α = 0.05): 0.0986
- Required sample size for 0.80 power: 1247

Cell Percentages:
Product     Product A  Product B  Product C  Row Total
Gender                                    
Male         18.18%    15.15%    12.12%     45.45%
Female       21.21%    18.18%    15.15%     54.55%
Col Total    39.39%    33.33%    27.27%    100.00%
```

Visualization Specifications:
1. Heat Map - Observed Frequencies:
   - Color range: [20, 35]
   - Color map: 'YlOrRd'
   - Annotation format: 'd' (integer)
   - Figure size: 12 x 5 inches
   - DPI: 100

2. Heat Map - Expected Frequencies:
   - Color range: [21.38, 33.85]
   - Color map: 'YlOrRd'
   - Annotation format: '.2f'
   - Figure size: 12 x 5 inches
   - DPI: 100

#### 3. Test of Homogeneity
Data File: homogeneity_test_data.csv
```
Group,Category,Count
Group1,Type A,45
Group1,Type B,35
Group1,Type C,20
Group2,Type A,40
Group2,Type B,40
Group2,Type C,20
Group3,Type A,35
Group3,Type B,45
Group3,Type C,20
```

Testing Steps:
1. Launch MinitabLikeApp.exe
2. Go to: File > Open (or use Ctrl+O)
3. Select and load homogeneity_test_data.csv
4. Navigate to: Stat > Advanced Statistics > Chi-Square Tests (or Alt+S, A, C)
5. Select "Test of Homogeneity" from the test type dialog
6. Variable Selection:
   - First prompt: Select "Category"
   - Second prompt: Select "Group"

Expected Results:
The session window should show exactly:
```
Chi-Square Test of Homogeneity Results

Test Information:
Category variable: Category
Group variable: Group
Number of categories: 3
Number of groups: 3

Test Statistics:
Chi-square statistic: 2.5000
Degrees of freedom: 4
p-value: 0.6446

Contingency Table:
Category    Type A  Type B  Type C
Group                             
Group1         45      35      20
Group2         40      40      20
Group3         35      45      20

Row Totals:
Group1    100
Group2    100
Group3    100

Column Totals:
Type A    120
Type B    120
Type C     60

Chi-Square Contributions:
Category    Type A    Type B    Type C
Group                                
Group1    0.6250    0.6250    0.0000
Group2    0.0000    0.0000    0.0000
Group3    0.6250    0.6250    0.0000

Decision:
Fail to reject the null hypothesis at α = 0.05

Note: The null hypothesis is that the distribution of categories is the same across groups.
```

Visualizations:
Three heat maps will appear showing:
1. "Observed Frequencies"
   - Shows actual counts (20-45)
   - Darker colors for higher values
2. "Expected Frequencies"
   - Shows calculated expected values
   - Uniform pattern if H0 is true
3. "Contributions to Chi-Square"
   - Shows individual cell contributions
   - Highlights cells with larger deviations

Additional Statistical Metrics:
```
Effect Size Measures:
- Cramer's V: 0.0645
- Contingency Coefficient: 0.0909

Standardized Residuals:
Category    Type A    Type B    Type C
Group                                
Group1     0.7906   -0.7906    0.0000
Group2     0.0000    0.0000    0.0000
Group3    -0.7906    0.7906    0.0000

Power Analysis:
- Observed Power (at α = 0.05): 0.1210
- Required sample size for 0.80 power: 1683

Cell Percentages:
Category    Type A    Type B    Type C    Row Total
Group                                    
Group1     15.00%    11.67%     6.67%     33.33%
Group2     13.33%    13.33%     6.67%     33.33%
Group3     11.67%    15.00%     6.67%     33.33%
Col Total  40.00%    40.00%    20.00%    100.00%

Expected Frequencies:
Category    Type A    Type B    Type C
Group                                
Group1      40.0      40.0      20.0
Group2      40.0      40.0      20.0
Group3      40.0      40.0      20.0
```

Visualization Specifications:
1. Heat Map - Observed Frequencies:
   - Color map: 'YlGnBu'
   - Annotation format: 'd'
   - Figure size: 15 x 5 inches
   - DPI: 100

2. Heat Map - Expected Frequencies:
   - Color map: 'YlGnBu'
   - Annotation format: '.1f'
   - Figure size: 15 x 5 inches
   - DPI: 100

3. Heat Map - Contributions:
   - Color map: 'YlOrRd'
   - Annotation format: '.4f'
   - Figure size: 15 x 5 inches
   - DPI: 100

Post-Hoc Analysis:
```
Pairwise Comparisons (Bonferroni-adjusted p-values):
Group Pairs    Chi-Square   p-value
Group1-Group2    0.6275     1.0000
Group1-Group3    2.5000     0.8595
Group2-Group3    0.6275     1.0000

Category Pairs   Chi-Square   p-value
TypeA-TypeB      2.5000      0.8595
TypeA-TypeC      0.4196      1.0000
TypeB-TypeC      0.4196      1.0000
```

Navigation and Shortcuts:
- Ctrl+O: Open file
- Alt+S: Open Stat menu
- Alt+W: Switch between windows
- Ctrl+C: Copy selected data
- F1: Help documentation
- Esc: Close current dialog

Specific Troubleshooting:
1. Data Structure Issues:
   - Group column must contain group identifiers
   - Category column must have consistent labels
   - Count column must be numeric
   - Check for duplicate combinations

2. Analysis Issues:
   - Minimum 5 expected counts per cell
   - At least 2 groups and 2 categories
   - No empty cells in contingency table

3. Display Issues:
   - Adjust window sizes for better view
   - Use "Arrange All" to organize plots
   - Save plots individually if needed
   - Export tables for detailed review

Data Export Options (Available for all Chi-Square Tests):
1. Results Export:
   - File > Save Results (or Ctrl+S)
   - Available formats:
     * CSV (comma-separated values)
     * Excel (.xlsx)
     * Text file (.txt)
   - Export options:
     * Complete results with all statistics
     * Contingency tables only
     * Summary statistics
     * Visualization data

2. Visualization Export:
   - Right-click on any plot
   - Export options:
     * PNG image (.png)
     * JPEG image (.jpg)
     * PDF document (.pdf)
   - Resolution settings:
     * Screen resolution (default)
     * High resolution (publication quality)
     * Custom DPI setting

Visualization Customization:
1. General Plot Controls:
   - Zoom: Use mouse wheel or zoom controls
   - Pan: Click and drag
   - Reset: Double-click or use reset button
   - Window size: Drag corners to resize

2. Color Scheme Options:
   - Right-click > Color Options
   - Available schemes:
     * Default (Blue/Orange)
     * Colorblind friendly
     * Grayscale
     * Custom colors

3. Label Customization:
   - Right-click > Edit Labels
   - Adjustable elements:
     * Axis titles
     * Data labels
     * Legend text
     * Title text
     * Font size and style

Statistical Interpretation Guidelines:
1. Chi-Square Goodness of Fit:
   - Null Hypothesis (H0): The data follows the specified distribution
   - Decision Rule:
     * If p-value < α (0.05), reject H0
     * If p-value ≥ α (0.05), fail to reject H0
   - Effect Size:
     * Small: χ² < 0.1
     * Medium: 0.1 ≤ χ² < 0.3
     * Large: χ² ≥ 0.3

2. Test of Independence:
   - Null Hypothesis (H0): The variables are independent
   - Key Considerations:
     * Expected frequencies ≥ 5
     * Degrees of freedom = (r-1)(c-1)
     * r = number of rows, c = number of columns
   - Interpretation:
     * p-value < 0.05: Evidence of association
     * Cramer's V for effect size:
       - Weak: V < 0.1
       - Moderate: 0.1 ≤ V < 0.3
       - Strong: V ≥ 0.3

3. Test of Homogeneity:
   - Null Hypothesis (H0): The proportions are the same across groups
   - Analysis Focus:
     * Compare distribution patterns
     * Examine standardized residuals
     * Look for systematic differences
   - Effect Assessment:
     * Standardized residuals > |2|: Significant difference
     * Pattern analysis in heat maps
     * Group proportion comparisons

### 4. Control Charts
Use control_chart_data.csv

#### X-bar R Chart
- Go to: Quality > Control Charts > X-bar R Chart
- Load all samples
- Expected Results:
  - X-bar chart with control limits
  - R chart with control limits
  - Out-of-control points highlighted
  - Process capability indices

### 5. Design of Experiments (DOE)
Refer to doe_test_cases.md for detailed test cases

#### Key Validations
1. Fractional Factorial:
   - Design matrix generation
   - Resolution calculation
   - Aliasing pattern display
   - Randomization

2. Response Surface:
   - Design point calculation
   - Center point placement
   - Alpha value computation
   - Block assignment

## Additional Features

### Random Data Generation
Test each distribution:

1. Normal Distribution
```
Settings:
Mean: 100
StdDev: 15
n: 100
```

2. Uniform Distribution
```
Settings:
Min: 0
Max: 100
n: 100
```

3. Binomial Distribution
```
Settings:
n: 10
p: 0.5
samples: 100
```

### File Operations
1. Test Save Function:
   - Generate some results
   - Save as CSV
   - Save as Excel
   - Verify file contents

2. Test Load Function:
   - Load each sample data file
   - Verify column recognition
   - Check data display

## Troubleshooting Guide
1. Data Import Issues:
   - Check file format (CSV, Excel)
   - Verify column headers
   - Check for special characters

2. Calculation Errors:
   - Verify input data ranges
   - Check for missing values
   - Confirm proper column selection

3. Display Issues:
   - Resize windows if needed
   - Check for graph rendering
   - Verify table formatting

## Test Sequence
1. Start with basic statistics
2. Progress to control charts
3. Test DOE functions
4. Perform Chi-Square analyses
5. Test process capability
6. Verify random data generation
7. Test file operations

## Notes
- Document any unexpected behavior
- Save results at each step
- Compare outputs with expected values
- Verify visualizations
- Check session window messages 