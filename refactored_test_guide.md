# Refactored Minitab-Like Application Test Guide

## Table of Contents
1. Basic Statistics
2. Advanced Statistics
3. Design of Experiments (DOE)
4. Quality Control
5. Six Sigma Tools
6. Random Data Generation
7. File Operations

## Menu Structure
1. File
   - Open
   - Save
   - Save As
   - Exit

2. Stat
   - Basic Statistics
     * Descriptive Statistics
     * Correlation
   - Advanced Statistics
     * Hypothesis Testing
     * ANOVA
     * Regression Analysis
     * Chi-Square Tests
   - Create DOE
   - Analyze DOE

3. Quality
   - Quality Tools
     * Pareto Chart
     * Probability Analysis
     * Process Capability
   - Control Charts
     * X-bar R Chart
     * Individual Chart
     * Moving Range Chart
   - Measurement System Analysis
     * Gage R&R Study
     * Linearity Study
     * Bias Study
     * Stability Study

4. Six Sigma
   - DMAIC Tools
     * Pareto Chart
     * Fishbone Diagram
     * FMEA Template
   - Six Sigma Metrics
     * DPMO Calculator
     * Sigma Level Calculator
     * Process Yield Analysis

5. Calc
   - Random Data
     * Normal
     * Uniform
     * Binomial
     * Poisson
   - Probability Distributions
     * Poisson

## Test Environment Setup
**Note**: The sample datasets described in this guide are for reference purposes. The actual application may have different datasets loaded at the time of testing. The variable names and values shown in the expected results might differ from what you see in the application, but the functionality and structure should be the same.

1. Run `run_minitab_app.py`
2. Required test data files:
   - sample_data.csv (Basic Statistics, Hypothesis Testing)
   - control_chart_data.csv (Control Charts)
   - pareto_data.csv (Pareto Analysis)

## 1. Basic Statistics

### 1.1 Descriptive Statistics
Test Steps:
1. Navigation:
   - Go to: Stat > Basic Statistics > Descriptive Statistics
2. Data Selection:
   - Load sample_data.csv
   - Select a numeric column (e.g., Height)
3. Dialog Options:
   - Check "Basic Statistics"
   - Check "Percentiles"
   - Check "Histogram"
   - Check "Box Plot"
4. Click "OK"

Expected Results:
- Session Window Output:
  ```
  Descriptive Statistics for column 'Height'
  Sample Size: 30

  Basic Statistics:
  Mean: 172.2900
  Median: 172.1500
  Mode: 169.9000
  Standard Deviation: 5.1327
  Variance: 26.3451
  Minimum: 162.4000
  Maximum: 181.7000
  Range: 19.3000
  First Quartile (Q1): 169.5000
  Third Quartile (Q3): 175.8250
  Interquartile Range (IQR): 6.3250
  Skewness: -0.0746
  Kurtosis: -0.7387

  Percentiles:
  1th Percentile: 162.6900
  5th Percentile: 163.8950
  10th Percentile: 165.4900
  25th Percentile: 169.5000
  50th Percentile: 172.1500
  75th Percentile: 175.8250
  90th Percentile: 179.2000
  95th Percentile: 180.2650
  99th Percentile: 181.3230
  ```
- Dialog Windows:
  - A histogram dialog with normal distribution fit
  - A box plot dialog displaying the data distribution

Error Handling:
- Non-numeric columns should be disabled in selection
- Empty data should display appropriate error message
- If no statistics options are selected, show a warning

### 1.2 Correlation Analysis
Test Steps:
1. Navigation:
   - Go to: Stat > Basic Statistics > Correlation
2. Data Selection:
   - Load sample_data.csv
   - Select two or more numeric columns (e.g., Height, Weight)
3. Click "OK"

Expected Results:
- Session Window Output:
  ```
  Correlation Analysis Results
  ---------------------------
  Correlations (Pearson):

           Height    Weight   
  Height    1.000    0.599**  
  Weight    0.599**  1.000    

  ** Correlation is significant at the 0.01 level
  ```
- Dialog Window:
  - A heatmap visualization of the correlation matrix

Error Handling:
- Non-numeric columns should be disabled
- If fewer than 2 columns are selected, show a warning
- Missing values should be handled appropriately

## 2. Advanced Statistics

### 2.1 Hypothesis Testing
Test Steps for One Sample t-test:
1. Navigation:
   - Go to: Stat > Advanced Statistics > Hypothesis Testing
2. Data Selection:
   - Load sample_data.csv
3. Dialog Options:
   - Select "One Sample t-test" radio button
   - Select a column (e.g., Height)
   - Enter a hypothesized mean (e.g., 170)
   - Select the alternative hypothesis (defaults to "≠ (Two-sided)")
   - Click "OK"

Note: The significance level (α) is fixed at 0.05 and not exposed in the UI.

Expected Results for One Sample t-test:
- Session Window Output:
  ```
  One Sample t-test
  Column: Height
  Hypothesized Mean (μ): 172.0
  Alternative Hypothesis: two-sided
  Sample Size: 30
  Sample Mean: 172.2900
  Sample Std Dev: 5.1327
  t-statistic: 0.3095
  p-value: 0.7592
  
  Conclusion: Fail to reject the null hypothesis.
  ```
- Dialog Window:
  - A histogram showing the data distribution with the hypothesized mean marked

Test Steps for Two Sample t-test:
1. In the same dialog, select "Two Sample t-test" radio button
2. Select two columns (e.g., Weight for males and females)
3. Check "Assume equal variances"
4. Click "OK"

Expected Results for Two Sample t-test:
- Session Window Output:
  ```
  Two Sample t-test
  First Column: Height
  Second Column: Weight
  Equal Variances: Yes
  Alternative Hypothesis: two-sided
  Sample 1 Size: 30
  Sample 1 Mean: 172.2900
  Sample 1 Std Dev: 5.1327
  Sample 2 Size: 30
  Sample 2 Mean: 65.1633
  Sample 2 Std Dev: 6.3848
  t-statistic: 71.6249
  p-value: 0.0000
  
  Conclusion: Reject the null hypothesis.
  ```
- Dialog Window:
  - A box plot comparing the two samples

Test Steps for Paired t-test:
1. In the same dialog, select "Paired t-test" radio button
2. Select "Height" for the "Before Column" dropdown
3. Select "Weight" for the "After Column" dropdown 
4. Select the alternative hypothesis (defaults to "≠ (Two-sided)")
5. Click "OK"

Expected Results for Paired t-test:
- Session Window Output:
  ```
  Paired t-test
  Before Column: Height
  After Column: Weight
  Alternative Hypothesis: two-sided
  Sample Size: 30
  Mean Difference: -107.1267
  Std Dev of Difference: 5.2783
  t-statistic: 111.1650
  p-value: 0.0000
  
  Conclusion: Reject the null hypothesis.
  ```
- Dialog Window:
  - Two plots showing:
    * A boxplot comparing before and after values
    * A histogram of differences between paired values

Error Handling:
- Non-numeric columns should be disabled
- Appropriate error messages for invalid inputs
- Warning if paired samples have different lengths

### 2.2 ANOVA
Test Steps for One-Way ANOVA:
1. Navigation:
   - Go to: Stat > Advanced Statistics > ANOVA
2. Data Selection:
   - Load `sample_data/one_way_anova_data.csv` containing columns like "Value", "Group" (categorical)
3. Dialog Options:
   - Select "One-Way ANOVA" radio button
   - Select "Value" as the response variable
   - Select "Group" as the factor
   - Click "OK"

Expected Results for One-Way ANOVA:
- Session Window Output:
  ```
One-Way ANOVA Results
Response Variable: Value
Factor: Group

ANOVA Table:
              sum_sq    df          F        PR(>F)
factor    290.762373   2.0  40.975636  2.893768e-13
Residual  308.675214  87.0        NaN           NaN

Group Statistics:
Means:
factor
Group1     9.623706
Group2    11.757675
Group3    14.025770
Name: response, dtype: float64
Standard Deviations:
factor
Group1    1.800013
Group2    1.862204
Group3    1.983966
Name: response, dtype: float64
Sample Sizes:
factor
Group1    30
Group2    30
Group3    30
Name: response, dtype: int64

Effect Size (η²): 0.4851
R-squared: 0.4851

Tukey's HSD Post-hoc Test:
Multiple Comparison of Means - Tukey HSD, FWER=0.05
==================================================
group1 group2 meandiff p-adj  lower  upper  reject
--------------------------------------------------
Group1 Group2    2.134 0.0001 0.9743 3.2937   True
Group1 Group3   4.4021    0.0 3.2424 5.5617   True
Group2 Group3   2.2681    0.0 1.1084 3.4278   True
--------------------------------------------------

  Conclusion: Reject the null hypothesis. There are significant differences between groups.
  ```
- Dialog Window:
  - A box plot showing the distribution of Value for each Group

Test Steps for Two-Way ANOVA:
1. In the same dialog, select "Two-Way ANOVA" radio button
2. Select "Response" as the response variable from `sample_data/two_way_anova_data.csv`
3. Select "Factor_A" as the first factor
4. Select "Factor_B" as the second factor
5. Click "OK"

Expected Results for Two-Way ANOVA:
- Session Window Output:
  ```
Two-Way ANOVA Results
Response Variable: Response
Factor A: Factor_A
Factor B: Factor_B

ANOVA Table:
                       sum_sq    df           F        PR(>F)
factor_a           158.907700   1.0  243.466757  4.585551e-14
factor_b           130.887423   2.0  100.268068  2.223822e-12
factor_a:factor_b   11.877148   2.0    9.098649  1.145827e-03
Residual            15.664499  24.0         NaN           NaN

Group Means:
factor_b         B1         B2         B3
factor_a                                 
A1        10.459003  12.437119  14.134923
A2        13.283760  17.867654  19.688659

Effect Sizes (Partial η²):
factor_a: 0.9103
factor_b: 0.8931
factor_a:factor_b: 0.4312

R-squared: 0.9506
Adjusted R-squared: 0.9404

  Conclusion: 
  - Significant main effect of Factor A
  - Significant main effect of Factor B
  - Significant interaction between Factor A and Factor B
  ```
- Dialog Window:
  - An interaction plot and Box plot showing the mean Response for each combination of Factor A and Factor B

Error Handling:
- Non-numeric response variables should be disabled
- Non-categorical factors should prompt a warning
- At least 2 groups required for one-way ANOVA
- At least 2 levels per factor required for two-way ANOVA
- Alert if insufficient data in any treatment combination for two-way ANOVA

### 2.3 Regression Analysis
Test Steps for Simple Linear Regression:
1. Navigation:
   - Go to: Stat > Advanced Statistics > Regression Analysis
2. Data Selection:
   - Load `sample_data/regression_data.csv` containing columns like "Height" and "Weight"
3. Dialog Options:
   - Select "Simple Linear Regression" radio button
   - From the "Response Variable (Y)" dropdown, select "X"
   - From the "Predictor Variable (X)" dropdown, select "Y"
   - Click "OK"

Expected Results for Simple Linear Regression:
- Session Window Output:
  ```
  Simple Linear Regression
   Response Variable: Y
   Predictor Variable: X

   Model Summary:
   R-squared = 0.9754
   Adjusted R-squared = 0.9749
   Root MSE = 0.9084
   Number of observations = 50

   Coefficients:
   Term        Coefficient   Std Error   t-value     p-value
   ------------------------------------------------------------
   Constant        1.0644      0.2531      4.2053  1.1313e-04
   X               1.9420      0.0436     44.5215  0.0000e+00

   Regression Equation:
   Y = 1.0644 + 1.9420×X

   Analysis of Variance:
   Source      DF          SS          MS           F         P
   ----------------------------------------------------------------------
   Regression   1    1635.5685    1635.5685    1902.8799  0.0000e+00
   Residual    48      41.2571       0.8595
   Total       49    1676.8256

  Conclusion: There is a significant linear relationship between Y and X.
  ```
- Dialog Window:
  - A scatter plot with the fitted regression line
  - Residual plots (residuals vs. fitted, normal probability plot)

Test Steps for Multiple Linear Regression:
Data Selection:
   - Load `sample_data/multiple_regression.csv`
1. In the same dialog, select "Multiple Linear Regression" radio button
2. Select "Sales" as the response variable
3. Select "Advertising", "Price", and "Promotion" as the predictor variables
4. Click "OK"

Expected Results for Multiple Linear Regression:
- Session Window Output:
  ```
  Multiple Linear Regression
   Response Variable: Sales
   Predictor Variables: Advertising, Price, Promotion

   Model Summary:
   R-squared = 0.9850
   Adjusted R-squared = 0.9775
   Root MSE = 1.1966
   Number of observations = 10
   Number of predictors = 3

   Coefficients:
   Term        Coefficient   Std Error   t-value     p-value    VIF
   ----------------------------------------------------------------------
   Constant      225.8219     27.8987      8.0943  1.9064e-04    -
   Advertising     1.9931      0.5389      3.6982  1.0110e-02     9.68
   Price         -12.4795      2.2436     -5.5622  1.4299e-03     7.63
   Promotion      -0.9606      1.0759     -0.8928  4.0633e-01     1.94

   Regression Equation:
   Sales = 225.8219 + 1.9931×Advertising - 12.4795×Price - 0.9606×Promotion

   Analysis of Variance:
   Source      DF    SS           MS           F-value     p-value
   ----------------------------------------------------------------------
   Regression    3     941.6014     313.8671     131.5217  7.3102e-06
   Residual      6      14.3186       2.3864
   Total         9     955.9200
  Conclusion: The regression model is statistically significant.
  Significant predictors: Advertising, Price
  ```
- Dialog Window:
  - Residual plot (residuals vs. fitted values)
  - Normal Q-Q plot
  - Histogram of residuals
  - VIF (Variance Inflation Factor) plot

Error Handling:
- Non-numeric variables should be disabled
- Warning if predictor variables are highly correlated
- Alert if insufficient observations for the number of predictors
- Error handling for perfect multicollinearity

### 2.4 Chi-Square Tests

**Data Preparation Note**: 
Before running chi-square tests, ensure your dataset contains properly formatted categorical variables. The dataset should have:
- Clear categorical columns (like Gender, Response, Category, etc.)
- Sufficient number of observations in each category
- No missing values in the categorical variables being tested

Test Steps for Chi-Square Test of Independence:
1. Navigation:
   - Go to: Stat > Advanced Statistics > Chi-Square Tests
2. Data Selection:
   - Load  \sample_data\contingency_data.csv
   - If you don't have appropriate data, create a new dataset with categorical columns:
     * Create columns like "Gender" (Male/Female) and "Response" (Yes/No/Maybe)
     * Add at least 20-30 rows with different combinations
3. Dialog Options:
   - Select "Test of Independence" radio button (usually the default option)
   - From the first dropdown, select your first categorical variable (e.g., "Gender")
   - From the second dropdown, select your second categorical variable (e.g., "Response")
   - You can optionally check "Create visualizations" if available
   - Click "OK" to perform the test

Expected Results for Test of Independence:
- Session Window Output:
  ```
   Column names:
   - Gender
   - Response

   Chi-Square Test of Independence
   ==============================

   First Variable: Gender
   Second Variable: Response

   Contingency Table:
   Response  Maybe  No  Yes
   Gender                  
   Female       30  48   52
   Male         20  55   45

   Chi-Square Statistic: 2.5850
   Degrees of Freedom: 2
   P-value: 0.2746

   Expected Frequencies:
   Response  Maybe     No    Yes
   Gender                       
   Female     26.0  53.56  50.44
   Male       24.0  49.44  46.56

   Conclusion: Fail to reject the null hypothesis.
   There is no evidence of a significant association between the variables.
  ```
- Dialog Window:
  - A heatmap visualization showing observed, expected, and contribution values
  
  **Note**: If you see a blank figure or visualization issues:
  - Check that your matplotlib backend is properly configured for your environment
  - Ensure your application has the necessary permissions to create new windows
  - Try resizing the dialog window to refresh the visualization
  - In some environments, you may need to install additional packages like 'python-tk' for proper GUI rendering

Test Steps for Goodness of Fit Test:
1. Navigation:
   - Go to: Stat > Advanced Statistics > Chi-Square Tests
2. Data Selection:
   - Load goodness_of_fit_data.csv
   - Load a dataset with at least one categorical variable (e.g., colors, categories, ratings)
   - A column with different category names like Red, Blue, Green, Yellow, Purple
3. Dialog Options:
   - Select "Goodness of Fit" radio button
   - From the first dropdown, select your categorical variable (e.g., "Category")
   - Note that when "Goodness of Fit" is selected, the second variable field is hidden as it's not needed
   - Check "Create visualizations" to see a bar chart of the results
   - Click "OK" to perform the test

Expected Results for Goodness of Fit Test:
- Session Window Output:
  ```
   Chi-Square Goodness of Fit Test
   =============================

   Variable: Category

   Chi-Square Statistic: 0.0000
   Degrees of Freedom: 4
   P-value: 1.0000


   Observed Frequencies:
   --------------------
   Red: 1
   Blue: 1
   Green: 1
   Yellow: 1
   Purple: 1

   Conclusion: Fail to reject the null hypothesis.
   The observed frequencies do not differ significantly from expected.
  ```
- Dialog Window:
  - A bar chart comparing observed vs. expected frequencies for each category

Test Steps for Test of Homogeneity:
1. Navigation:
   - Go to: Stat > Advanced Statistics > Chi-Square Tests
2. Data Selection:
   - Load homogeneity_data.csv
   - Load a dataset with at least two categorical variables
   - For example, a dataset with columns like "Location" and "Defect_Type"
3. Dialog Options:
   - Select "Test of Homogeneity" radio button
   - From the first dropdown, select your first categorical variable (e.g., "Location")
   - From the second dropdown, select your second categorical variable (e.g., "Defect_Type")
   - Check "Create visualizations" to see visual representations of the data
   - Click "OK" to perform the test

Expected Results for Test of Homogeneity:
- Session Window Output:
  ```
   Column names:
   - Location
   - Defect_Type
   - Count

   Chi-Square Test of Homogeneity
   ==============================

   First Variable: Location
   Second Variable: Defect_Type

   Contingency Table:
   Defect_Type  Critical  Major  Minor
   Location                           
   Plant_A             1      1      1
   Plant_B             1      1      1
   Plant_C             1      1      1

   Chi-Square Statistic: 0.0000
   Degrees of Freedom: 4
   P-value: 1.0000

   Expected Frequencies:
   Defect_Type  Critical  Major  Minor
   Location                           
   Plant_A           1.0    1.0    1.0
   Plant_B           1.0    1.0    1.0
   Plant_C           1.0    1.0    1.0

   WARNING: Some expected frequencies are less than 5. Chi-square test may not be reliable.

   Conclusion: Fail to reject the null hypothesis.
   There is no evidence that the proportions differ across groups.
  ```
- Dialog Window:
  - A heatmap visualization showing observed, expected, and contribution values

Error Handling:
- Non-categorical variables should prompt a warning
- Expected frequencies less than 5 should trigger a warning message
- Error if insufficient data in contingency table cells
- **Warning message "Not enough data points for chi-square test" appears if there are insufficient observations in the dataset**
- "Error in Chi-Square Tests" message if there are issues with the calculation or visualization

#### Troubleshooting Chi-Square Test Issues

If you encounter the "Not enough data points for chi-square test" warning:

1. **Check your dataset structure**:
   - Ensure your dataset contains proper categorical variables (text or categorical codes)
   - Verify that your dataset isn't mixed with numerical data in the same columns
   - Make sure categorical columns don't have too many missing values

2. **Data preparation steps**:
   - If using numerical data that should be treated as categories, convert them to categorical type first
   - Filter out any rows with missing values in the categorical columns
   - Ensure you have adequate sample size (typically at least 5 expected counts per cell)

3. **Alternative approach**:
   - For small sample sizes, consider using Fisher's exact test (though not currently implemented)
   - Collapse categories if appropriate to increase counts per cell

If you experience visualization issues:
1. Ensure the "Create visualizations" checkbox is selected
2. Wait a few seconds for the visualization to render
3. Try adjusting the window size if the visualization appears blank
4. Check your system's matplotlib configuration

If you're unsure which test to use:
- **Test of Independence**: Use when you want to determine if there's a significant association between two categorical variables
- **Goodness of Fit**: Use when you want to test if a single categorical variable follows an expected distribution
- **Test of Homogeneity**: Use when you want to test if different populations or groups have the same distribution of a categorical variable

### 2.5 Design of Experiments (DOE)

#### 2.5.1 2-level Factorial Design
Test Steps for Full Factorial Design:
1. Navigation:
   - Go to: Stat > Create DOE
2. Dialog Options:
   - Select "2-level Factorial"
3. Design Setup:
   ```
   Type: 2-level factorial
   Number of factors: 3
   ```
4. Factor Information:
   ```
   Factor  Name    Units   Low   High
   A       Temp    °C      150   180
   B       Press   MPa     2.0   3.0
   C       Time    min     30    45
   ```
   - Click "OK"

   Expected Results for 2-level Factorial Design:
   - Session Window Output:
   ```
   Full Factorial Design Created
   Number of Factors: 3
   Number of Runs: 8

   Factors and Levels:
   Temp: 150 | 180
   Press: 2 | 3
   Time: 30 | 45
   ```
- The main table should display the experimental design with:
  * Run order (1, 2, 3, etc.)
  * Factor columns with coded values (-1, 1)
  * Additional columns with actual values based on specified low/high levels
  * A "Response" column (empty) for entering experimental results

StdOrder	RunOrder	Temp	Press	Time	Temp_actual	Press_actual	Time_actual	Response
   1	      3	      -1	   -1	   -1	   150	      2	            30	      nan
   2	      5	      -1	   -1	   1	   150	      2	            45	      nan
   3	      8	      -1	   1	   -1	   150	      3	            30	      nan
   4	      6	      -1	   1	   1	   150	      3	            45	      nan
   5	      7	      1	   -1	   -1	   180	      2	            30	      nan
   6	      2	      1	   -1	   1	   180	      2	            45	      nan
   7	      4	      1	   1	   -1	   180	      3	            30	      nan
   8	      1	      1	   1	   1	   180	      3	            45	      nan

5. Verification Steps:
   - Check design properties:
     * Resolution
     * Alias structure
     * Design generators
   - Verify randomization
   - Check factor level combinations

6. Error Handling:
   - Invalid factor levels (should show error)
   - Too many factors (should warn)
   - Insufficient runs (should warn)

### 2.6  Analyze DOE
   Sample Data (from doe_with_responses.csv):
   Temperature,Pressure,Response,Temperature_actual,Pressure_actual,Press_actual,Time_actual
   0,-1.41,45,165,2.5,2.5,37.5
   -1.41,0,65,165,2.5,2,30
   0,0,55,165,2.5,3,45
   -1,1,45,150,3,2.5,37.5
   1,-1,65,180,2,3,30
   0,1.41,55,165,2.5,2.5,37.5
   -1,-1,45,150,2,3,45
   0,0,65,165,2.5,2,45
   1.41,0,55,165,2.5,2.5,37.5
   0,0,45,165,2.5,2,30
   0,0,65,165,2.5,2.5,37.5
   0,0,55,165,2.5,3,30
   1,1,45,180,3,2.5,37.5
   -1,-1,65,156,150,2,45
   ```

   Test Steps:
   1. Navigation:
      - Go to: Stat > Analyze DOE

   2. Data Selection:
      - Load doe_with_responses.csv   

   3. Expected Results:
      ```
      Design of Experiments Analysis

      Model Summary:
      R-squared: 0.9997
      Adjusted R-squared: 0.9995
      F-statistic: 4494.9707
      Prob(F-statistic): 0.0000

      Effects:
      Temperature: 462.0498
      Pressure: 36.5857
      Time: 35.4059

      Analysis of Variance:
      ==============================================================================
      Omnibus:                        1.161   Durbin-Watson:                   1.631
      Prob(Omnibus):                  0.560   Jarque-Bera (JB):                0.616
      Skew:                           0.055   Prob(JB):                        0.735
      Kurtosis:                       1.645   Cond. No.                         3.23
      ==============================================================================

      Parameter Estimates:
      ===============================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
      -------------------------------------------------------------------------------
      const        1733.0483      2.458    704.985      0.000    1726.223    1739.874
      Temperature   231.0249      2.458     93.978      0.000     224.200     237.850
      Pressure       18.2929      2.458      7.441      0.002      11.468      25.118
      Time           17.7030      2.458      7.201      0.002      10.878      24.528
      ===============================================================================
      ```
4. Dialog Window:
   - A Normal Probability Plot of Effects, and A Main Effects Plot

5. Verification Steps:
   - Check design properties:
     * Resolution level
     * Number of runs
     * Design generator
   - Verify factor levels
   - Confirm fraction size

6. Error Handling:
   - Invalid factor levels (should show error)
   - Too many factors (should warn)
   - Insufficient runs (should warn)


Test Steps for Response Surface Design:
1. Navigation:
   - Go to: Stat > Create DOE
2. Dialog Options:
   - Select "Response Surface Design (Central Composite)" radio button
   - Add 2-3 factors following the same procedure as above
   - Set "Center Points" to 5
   - Set "Alpha Value" to 1.414 (default)
   - Click "OK"

Expected Results for Response Surface Design:
- Session Window Output:
  ```
  Creating Response Surface Design (Central Composite)
  Number of Factors: 2
  Factors: Temperature, Pressure
  Center Points: 5
  Alpha: 1.414

  Design Summary:
  Total Runs: 13
  Randomized: Yes
  ```
- The main table should display a central composite design including:
  * Factorial points (-1, 1)
  * Axial points (-1.414, 1.414)
  * Center points (0, 0)

Error Handling for Create DOE:
- Warning if fewer than 2 factors are provided
- Warning for invalid input values (non-numeric low/high levels)
- Warning for invalid fraction in fractional designs
- Auto-correction of missing factor names

#### Adding Response Values to the DOE Design
After creating your DOE design, you need to enter response values before analysis:

1. In the main data table, locate the "Response" column (typically the rightmost column)
2. To enter a value:
   - Click on a cell in the "Response" column that shows "nan" (Not a Number)
   - Type a numeric value
      Type "=45" (with the equals sign) and press Enter
      In many applications, the equals sign forces numeric interpretation
   - Press Enter or Tab to confirm and move to the next cell

3. Continue entering values for each run in the experiment
4. For testing purposes, you can follow these patterns for realistic test values:
   - For a 2-factor design: Enter higher values (e.g., 80-100) when both factors are at the same level (both high or both low), and lower values (e.g., 40-60) when factors are at opposite levels
   - For a 3-factor design: Enter higher values (e.g., 90-100) when all factors are at high level (+1), medium values (e.g., 50-70) for mixed levels, and lower values (e.g., 30-50) when all factors are at low level (-1)

**Example Response Values for a 2³ Factorial Design:**
| Run | Temperature | Pressure | Time | Response |
|-----|-------------|----------|------|----------|
| 1   | -1          | -1       | -1   | 45       |
| 2   | 1           | -1       | -1   | 65       |
| 3   | -1          | 1        | -1   | 55       |
| 4   | 1           | 1        | -1   | 75       |
| 5   | -1          | -1       | 1    | 50       |
| 6   | 1           | -1       | 1    | 70       |
| 7   | -1          | 1        | 1    | 60       |
| 8   | 1           | 1        | 1    | 95       |

**Note:** It's important to enter response values before proceeding to the "Analyze DOE" step. The analysis will not produce meaningful results without response data.

#### Saving DOE Data
After entering your response values, you should save your design to preserve your work:

1. With the DOE data displayed in the main table, go to: File > Save (or File > Save As for a new file)
2. If this is a new design without a file path, a dialog will appear:
   - Choose a location to save the file
   - Enter a filename (e.g., "my_doe_experiment.csv")
   - Select the file type (.csv recommended)
   - Click "Save"
3. The data will be saved with all your factors, coded levels, actual values, and response values
4. You'll see a confirmation message in the session window: "Saved to file: [file path]"

**Important:** The application automatically updates the internal data when you edit cell values, but saving to a file requires using the Save or Save As function. If you close the application without saving, your response values will be lost.

#### 2.5.2 Analyze DOE
Prerequisite: Create a DOE design using the steps above and enter response values in the "Response" column. For testing purposes, you can enter random values or use the following pattern:
- For a 2-factor design: Higher values when both factors are at the same level (both high or both low)
- For a 3-factor design: Higher values when all factors are at high level

Test Steps:
1. Navigation:
   - Go to: Stat > Analyze DOE
2. Dialog Options:
   - Ensure all three analysis types are checked (default):
     * Main Effects
     * Interaction Effects
     * Normal Plot of Effects
   - Verify "Response" is selected in the "Response Column" dropdown
   - Set "Significance Level" to 0.05 (default)
   - Click "OK"

Expected Results:
- Session Window Output:
  ```
  Design of Experiments Analysis
  Response Variable: Response
  Factors: Temperature, Pressure, Time

  Effects:
  ------------------------------
  Term      Effect
  ------------------------------
  Temperature 3.2500
  Pressure    1.7500
  Time        0.7500
  Temperature:Pressure 2.5000
  Temperature:Time     0.2500
  Pressure:Time       -0.5000
  ```
- Dialog Windows:
  - A main effects plot showing the effect of each factor on the response
  - An interaction effects plot showing how factors interact
  - A normal probability plot of effects to identify significant factors

Error Handling for Analyze DOE:
- Warning if no data is available
- Warning if the Response column is missing or contains no values
- Warning if too many response values are missing
- Warning if no factor columns are found

Additional Testing Considerations:
- Verify that designs with different numbers of factors work correctly
- Test the handling of missing response values
- Verify that the randomize option properly randomizes the run order
- Test the conversion between coded values (-1, 1) and actual values

#### Troubleshooting DOE Analysis Issues

If you encounter problems with DOE analysis, here are some common issues and solutions:

1. **"Response column is missing or contains no values" warning:**
   - Check that you have entered numeric values in the Response column
   - Ensure you don't have any text entries in the Response column (only numbers are accepted)
   - Make sure you've clicked outside the cell or pressed Enter after typing the last value

2. **Incorrect or unexpected analysis results:**
   - Verify that your response values follow a logical pattern for testing (e.g., higher when factors are at optimal levels)
   - Check that you didn't accidentally enter response values in the wrong rows
   - Ensure factor levels (-1, 1) haven't been accidentally modified

3. **Missing values in the analysis:**
   - The analysis can handle some missing values (up to 20% of response values)
   - If more than 20% are missing, you'll receive a warning and results may not be reliable
   - For best results, enter values for all experimental runs

4. **Visualization issues:**
   - If plots appear blank or incorrectly sized, try resizing the dialog window
   - Close and reopen the analysis dialog if visualizations don't display properly
   - Check that you have at least 2 factors if you want to see interaction effects

5. **Data entry tips:**
   - You can use Tab to move to the next cell after entering a value
   - To replace "nan" with zero, you can simply enter "0"
   - The software will attempt to convert text entries to numbers when possible

## 3. Quality Control

### 3.1 Pareto Chart
Test Steps:
1. Navigation:
   - Go to: Six Sigma > DMAIC Tools > Pareto Chart
2. Data Entry:
   - Enter a title (e.g., "Product Defects") in the "Chart Title" field at the top
   - For each category and value:
     * Type the category name (e.g., "Scratches") in the "Category" field
     * Type the value (e.g., "35") in the "Value" field
     * Click the "Add" button to add this entry to the list
     * Repeat this process for each category-value pair:
       - Scratches: 35
       - Missing Parts: 22
       - Wrong Color: 15
       - Dents: 12
       - Other: 5
   - Verify all entries appear correctly in the list
   - Click "OK" to generate the chart

Expected Results:
- Session Window Output:
  ```
  Created Pareto Chart: Product Defects
  Categories: Scratches, Missing Parts, Wrong Color, Dents, Other
  Total Count: 89
  ```
- Dialog Window:
  - A Pareto chart showing categories in descending order
  - Cumulative percentage line overlay
  - Bars for each category

Error Handling:
- At least 2 categories required
- Values must be positive numbers
- Empty title should show a warning

### 3.2 Probability Analysis
Test Steps:
1. Navigation:
   - Go to: Quality > Quality Tools > Probability Analysis
2. Data Selection:
   - Load sample_data.csv or use existing data
   - The dialog should display only numeric columns; verify that non-numeric columns are not selectable
3. Dialog Options:
   - Select a numeric column (e.g., "Height")
   - Select Distribution: "Normal" (default)
   - Confidence Level: 0.95 (default)
   - Display Options (check all):
     * Histogram with Distribution Fit
     * Q-Q Plot
     * Cumulative Distribution Function (CDF)
4. Click "OK"

Expected Results:
- Session Window Output:
  ```
  Probability Analysis Results
  --------------------------
  Probability Analysis for column: Height
  --------------------------------------
  Sample Size: 30
  Mean: 172.2900
  Median: 172.1500
  Standard Deviation: 5.1327
  Minimum: 162.4000
  Maximum: 181.7000
  
  Fitted Distribution: Normal
  Parameters: μ=172.2900, σ=5.1327
  
  Percentiles:
  1th Percentile: 162.6900
  5th Percentile: 163.8950
  10th Percentile: 165.4900
  25th Percentile: 169.5000
  50th Percentile: 172.1500
  75th Percentile: 175.8250
  90th Percentile: 179.2000
  95th Percentile: 180.2650
  99th Percentile: 181.3230
  
  95.0% Confidence Interval for Mean:
  (170.3434, 174.2366)
  
  Normality Tests:
  Shapiro-Wilk Test:
    Statistic: 0.9923
    p-value: 0.9741
    Appears normal (α=0.05)
  
  D'Agostino's K² Test:
    Statistic: 0.7218
    p-value: 0.6971
    Appears normal (α=0.05)
  
  Goodness of Fit Test (Kolmogorov-Smirnov):
    Statistic: 0.0721
    p-value: 0.9951
    The data appears to follow a normal distribution (α=0.05)
  ```
- Visualization Windows:
  - A histogram with the fitted distribution curve (if selected)
  - A Q-Q plot comparing theoretical vs. sample quantiles (if selected)
  - A Cumulative Distribution Function (CDF) plot showing both empirical and theoretical CDFs (if selected)

Testing Different Distributions:
1. Repeat the above steps but select "Lognormal" as the distribution type
   - Expected Results: The output should show the lognormal distribution parameters (shape, location, scale)
   - The fitted curve should match a lognormal distribution

2. Repeat the above steps but select "Exponential" as the distribution type
   - Expected Results: The output should show the exponential distribution parameter (lambda)
   - The goodness of fit test may show a poorer fit for columns with data that isn't exponentially distributed

Error Handling:
- If no data is loaded, a warning message should appear
- If selected column contains non-numeric data, appropriate error should be shown
- If insufficient data (less than 5 data points), warning should be displayed
- If the data strongly violates the assumptions of the selected distribution, the goodness of fit test should show a low p-value

Verification Steps:
1. Check that the calculated statistics match the actual data properties
2. Verify that the fitted distribution parameters are correctly calculated
3. Confirm that the visual plots correctly represent the data and theoretical distributions
4. Ensure the goodness of fit test correctly identifies whether the data follows the specified distribution

Tips and Troubleshooting:
- For data that is known to follow a specific distribution, the goodness of fit test should give a high p-value
- For skewed data, try the lognormal distribution which often provides a better fit
- For data representing waiting times or intervals between events, try the exponential distribution
- If visualization windows appear but are blank, try resizing them to refresh the display
- The Shapiro-Wilk test is generally more powerful for small sample sizes (n<50) compared to other normality tests

### 3.3 Process Capability

Test Steps:
1. Navigation:
   - Go to: Quality > Quality Tools > Process Capability
2. Data Selection:
   - Load or use existing data with continuous measurements
   - Sample data: A dataset with process measurements like part dimensions or weights
3. Dialog Options:
   - Select a numeric column for analysis (e.g., "Diameter")
   - Enter specification limits:
     * Lower Specification Limit (LSL): e.g., 9.8
     * Upper Specification Limit (USL): e.g., 10.2
   - Optionally enter a Target Value: e.g., 10.0
   - Check/uncheck options as needed:
     * Use subgroups (if you have categorical columns)
     * Check normality
     * Display histogram
4. Click "OK"

Expected Results:
- Session Window Output:
  ```
  Process Capability Analysis for Diameter
  ----------------------------------------

  Basic Statistics:
  Sample Size: 50
  Mean: 10.0123
  Standard Deviation: 0.0531
  Minimum: 9.8975
  Maximum: 10.1354

  Specification Limits:
  Lower Specification Limit (LSL): 9.8000
  Upper Specification Limit (USL): 10.2000
  Target Value: 10.0000

  Capability Indices:
  Cp: 1.2565
  Cpk: 1.2432
  CPU: 1.1765
  CPL: 1.3365
  Cpm: 1.2487

  Performance Indices:
  Pp: 1.2565
  Ppk: 1.2432
  PPU: 1.1765
  PPL: 1.3365

  Process Capability Interpretation:
  The process is capable (Cpk ≥ 1.33)

  Normality Test (D'Agostino's K²):
  p-value: 0.7826
  The data appears to be normally distributed (p ≥ 0.05)
  ```
- If "Display histogram" is checked, a visualization window should appear showing:
  - A histogram of the data with normal curve overlay
  - Vertical lines for LSL, USL, and Target (if specified)
  - Box plot and Q-Q plot for assessing normality

Error Handling:
- If no data is loaded, a warning message should appear
- If selected column contains non-numeric data, appropriate error should be shown
- If insufficient data (less than 10 data points), warning should be displayed
- If LSL >= USL, error message should indicate invalid specification limits
- If data is non-normal and normality checking is enabled, a warning about reliability of indices should be shown

#### Testing with Subgroups

If your data includes categorical variables that define process subgroups (like "Shift", "Machine", or "Batch"), you can test the subgroup functionality:

1. Follow the same steps as above, but also:
   - Check "Use subgroups"
   - Select a categorical column from the "Subgroup Column" dropdown
   
2. The analysis should calculate capability indices both overall and within subgroups
   - Results should note if there are significant differences between subgroups
   - The capability interpretation should consider both within and overall variation

#### Troubleshooting

If you encounter issues with the Process Capability analysis:

1. **Specification limits aren't appropriate:**
   - The default specification limits are set to ±3 standard deviations from the mean
   - Adjust these to match your actual process requirements
   
2. **Warning about non-normal data:**
   - Process capability indices are based on the assumption of normality
   - If your data is significantly non-normal, consider:
     * Transforming the data (e.g., log transformation)
     * Using non-parametric methods

3. **Interpretation guidance:**
   - Cp/Cpk < 1.0: Process is not capable
   - 1.0 ≤ Cp/Cpk < 1.33: Process is marginally capable
   - Cp/Cpk ≥ 1.33: Process is capable
   - Cp/Cpk ≥ 1.67: Process is highly capable

### 3.4 Measurement System Analysis

#### 3.4.1 Gage R&R Study

Required Data:
- File: msa_data.csv
- Columns needed:
  * Part (text or numeric)
  * Operator (text)
  * Measurement (numeric)
  * Order (numeric, optional)
- Minimum requirements:
  * Parts: 5 minimum (10 recommended)
  * Operators: 2-10
  * Replicates: 2-5 per part-operator
- Data validation:
  * Measurements should be numeric
  * No missing values allowed
  * Balanced design preferred (same number of measurements for each part-operator combination)

Test Steps:
1. Data Loading:
   - Go to: File > Open
   - Select "sample_data/msa_data.csv"
   - Verify data types:
     * Part (text/numeric)
     * Operator (text)
     * Measurement (numeric, 2 decimals)
     * Order (numeric, if included)

2. Function Access:
   - Go to: Quality > Measurement System Analysis > Gage R&R
   - Verify dialog opens with all options available

3. Dialog Completion:
   a. Column Selection:
      - When prompted "Select Part Column", choose "Part"
      - When prompted "Select Operator Column", choose "Operator"
      - When prompted "Select Measurement Column", choose "Measurement"
      - When prompted "Select Order Column", choose "Order" (if using)
   
   b. Analysis Options:
      - Study Type: "Crossed" (default)
      - Process tolerance (optional): Check and enter 0.60
      - Confidence level: 95% (default)
      - Include Part * Operator Interaction: Check this option
   
   c. Visualization Options:
      - Components of Variation: Check this option
      - R Chart by Operator: Check this option
      - X-bar Chart by Operator: Check this option
      - Measurement by Part: Check this option
      - Measurement by Operator: Check this option
      - Part * Operator Interaction: Check this option
      - Run Chart (requires Order column): Uncheck if not using Order

4. Click "OK"

Expected Results:
1. Session Window Output:
   ```
   === Gage R&R Study Results ===
   
   Study Date: [current date/time]
   Data Source: [file path or "New Data"]
   
   Study Parameters:
     - Parts: 10
     - Operators: 3
     - Replicates: 3
     - Total Measurements: 90
     - Study Type: Crossed
     - Process Tolerance: 0.60
   
   === ANOVA Results ===
   [ANOVA table with source, DF, SS, MS, F-value, P-value]
   
   === Variance Components ===
   Source                Variance    % Contribution    % Study Var    % Tolerance
   -------------------------------------------------------------------------------------------
   Part-to-Part:         [value]     [80-99]%          [90-99]%       [60-90]%
   Total Gage R&R:       [value]     [1-20]%           [10-40]%       [5-30]%
     Repeatability:      [value]     [1-10]%           [10-30]%       [5-20]%
     Reproducibility:    [value]     [1-10]%           [5-20]%        [3-15]%
       Operator:         [value]     [0-5]%            [0-15]%        [0-10]%
       Interaction:      [value]     [0-5]%            [0-15]%        [0-10]%
   Total Variation:      [value]     100.00%           100.00%        100.00%
   
   Number of Distinct Categories: [5 or more]
   
   === Study Assessment ===
   Total Gage R&R: [less than 30]%
   Assessment: [Acceptable or Excellent] measurement system
   ```

2. Visualizations:
   - A series of plots should appear in a separate window, including:
     * Components of Variation bar chart
     * R Chart by Operator
     * X-bar Chart by Operator
     * Measurement by Part boxplot
     * Measurement by Operator boxplot
     * Part*Operator Interaction plot

Error Handling:
1. No Data Loaded:
   - If no data is loaded, a prompt should appear asking if you want to load sample data
   - Selecting "Yes" should load the msa_data.csv file
   - Selecting "No" should return to the main window without analysis

2. Data Validation Errors:
   - If selected columns contain missing values, an error message should appear
   - If fewer than 2 operators are selected, an error message should appear
   - If fewer than 5 parts are selected, a warning (not error) should appear
   - If the design is unbalanced, a warning should appear with option to continue

3. Interpretation Alerts:
   - If Gage R&R % is above 30%, the assessment should say "Unacceptable measurement system"
   - If Number of Distinct Categories is less than 5, a warning should appear

Troubleshooting:
1. Visualization Issues:
   - If plots appear blank or incorrectly sized, try resizing the dialog window
   - Close and reopen the analysis dialog if visualizations don't display properly

2. Study Assessment:
   - Gage R&R < 10%: Excellent measurement system
   - 10% ≤ Gage R&R < 30%: Acceptable measurement system
   - Gage R&R ≥ 30%: Unacceptable measurement system - needs improvement
   - Number of Distinct Categories should be ≥ 5 for an acceptable system

3. Design Requirements:
   - For valid results, the study should have at least:
     * 5 parts (10 recommended)
     * 2 operators (3 recommended)
     * 2 replicates per part-operator combination (3 recommended)
   - The total number of measurements should be at least 30

4. Working with Non-Ideal Data:
   - If your data is unbalanced, the analysis can still proceed but results may be less reliable
   - If you have fewer than recommended parts/operators, interpret results with caution
   - If the Part * Operator interaction is significant (p-value < 0.05), this suggests different operators measure parts differently

### 3.3 Linearity Study
Test Steps:
1. Navigation:
   - Go to: Quality > Measurement System Analysis > Linearity Study

2. Data Input:
   a. Using Sample Data:
   - When prompted "No data is currently loaded. Do you want to load the sample linearity data?", click "Yes"
   
   b. Using Existing Data:
   - If you already have data loaded, select:
     * Reference Column: column containing known standard values
     * Measurement Column: column containing actual measurements
     * Operator Column (optional): column identifying who made each measurement
     * Order Column (optional): column indicating measurement sequence

3. Dialog Options:
   a. Options Tab:
   - Confidence Level: 0.95 (default)
   - Include operator effects: checked
   - Fit intercept: checked
   - Include tolerance: unchecked by default
     * When checked, enter Tolerance value (e.g., 1.0)
   - Target Bias: 0.0 (default)
   
   b. Graphs Tab:
   - Linearity Plot: checked
   - Bias Plot: checked
   - Percent Bias Plot: checked
   - Fitted Line Plot: checked
   - Residual Plots: checked

4. Expected Results:
   ```
   === Linearity Study Results ===

   Study Date: [current date and time]
   Data Source: [file name or "New Data"]

   Study Parameters:
     - Reference Values: [5-10]
     - Operators: [2-3]
     - Total Measurements: [30-60]
     - Confidence Level: 0.95
     [- Tolerance: 1.000] (if specified)

   === Regression Statistics ===
   Slope: [value between -0.1 and 0.1 for a good system]
   Intercept: [value between -0.1 and 0.1 for a good system]
   R-squared: [value > 0.9 for a good system]
   Standard Error: [small value, typically < 0.05]

   === Acceptance Criteria ===
   Slope: [Acceptable/Not Acceptable]
   Intercept: [Acceptable/Not Acceptable]
   R-squared: [Acceptable/Not Acceptable]

   === Bias Analysis ===
   Average Bias: [small value, ideally close to 0]
   Maximum Bias: [larger value but ideally < 0.1]

   Reference Value | Bias | StdDev | n
   ----------------------------------
   [ref value 1] | [bias 1] | [stddev 1] | [count 1]
   [ref value 2] | [bias 2] | [stddev 2] | [count 2]
   ...

   [=== Tolerance Analysis ===] (if tolerance specified)
   [Average % of Tolerance: [value < 10% for a good system]]
   [Maximum % of Tolerance: [value < 30% for a good system]]
   [Overall Assessment: [Acceptable/Not Acceptable]]

   === Overall Assessment ===
   [The measurement system linearity is acceptable.]
   [OR: The measurement system linearity needs improvement.]
   [- The slope indicates non-linear behavior across the measurement range.] (if applicable)
   [- The intercept indicates a consistent bias in measurements.] (if applicable)
   [- The R-squared value indicates poor fit or high variability.] (if applicable)
   ```

5. Visualizations:
   - A series of plots should appear in a separate window, including:
     * Linearity Plot: Measurements vs Reference Values
     * Bias Plot: Bias vs Reference Values
     * Percent Bias Plot: Percent Bias vs Reference Values (if applicable)
     * Fitted Line Plot: Regression line with confidence bands
     * Residual Plots: Residuals vs Reference Values

Error Handling:
1. No Data Loaded:
   - If no data is loaded, a prompt should appear asking if you want to load sample data
   - Selecting "Yes" should load the linearity_data.csv file
   - Selecting "No" should return to the main window without analysis

2. Data Validation Errors:
   - If Reference and Measurement columns are the same, an error message should appear
   - If selected columns contain missing values, an error message should appear
   - If fewer than 3 reference values are detected, a warning message should appear

Troubleshooting:
1. Interpretation Guidelines:
   - A good measurement system should have:
     * Slope close to 0 (< 0.1 in absolute value)
     * Intercept close to 0 (< 0.1 in absolute value)
     * R-squared > 0.9
     * Percent of tolerance < 10% (if tolerance is specified)
   
2. Common Issues:
   - If the slope is significant (not close to 0), the measurement bias changes across the measurement range
   - If the intercept is significant (not close to 0), there is a consistent offset in measurements
   - If R-squared is low, there is high variability or poor fit in the measurements
   - If you get "Not Acceptable" results, check if measurement equipment needs calibration

3. Working with Non-Ideal Data:
   - Minimum 3 reference values are required, but 5 or more are recommended
   - Multiple measurements at each reference value improve statistical reliability
   - If operator effects are significant, consider operator training or standardization

### 3.4 Bias Study
Test Steps:
1. Navigation:
   - Go to: Quality > Measurement System Analysis > Bias Study

2. Data Input:
   a. Using Sample Data:
   - When prompted "No data is currently loaded. Do you want to load the sample bias data?", click "Yes"
   
   b. Using Existing Data:
   - If you already have data loaded, select:
     * Measurement Column: column containing actual measurements
     * Reference Value Option:
       - Use single reference value: enter a known standard value
       - Use reference column: select column containing known standard values
     * Operator Column (optional): column identifying who made each measurement

3. Dialog Options:
   a. Options Tab:
   - Confidence Level: Select "95%" (default)
   - Include operator effects: checked
   - Include tolerance: unchecked by default
     * When checked, enter Tolerance value (e.g., 1.0)
     * Enter Acceptable Bias %: 5.0% (default)
   
   b. Graphs Tab:
   - Histogram with Reference Value: checked
   - Run Chart: checked
   - Normal Probability Plot: checked
   - Operator Comparison Plot: checked

4. Expected Results:
   ```
   === Bias Study Results ===

   Study Date: [current date and time]
   Data Source: [file name or "New Data"]

   Study Parameters:
     - Measurement Column: Measurement
     - Reference Value: 10.5000 (or Reference Column: Reference)
     - Measurements: [30-60]
     - Operators: [2-3] (if applicable)
     - Confidence Level: 0.95
     [- Tolerance: 1.000] (if specified)
     [- Acceptable Bias: ±5.0%] (if tolerance specified)

   === Basic Statistics ===
   Number of Measurements: [30-60]
   Mean: [value close to reference value]
   Standard Deviation: [small value, typically < 0.05]
   Standard Error: [very small value]

   === Bias Analysis ===
   Absolute Bias: [small value, ideally < 0.05]
   Percent Bias: [small percentage, ideally < 5%]

   === Hypothesis Test (H₀: μ = [reference value]) ===
   t-statistic: [value]
   p-value: [value]
   Conclusion: [Significant bias detected / No significant bias]

   === 95% Confidence Interval for Bias ===
   Lower: [value]
   Upper: [value]

   [=== Tolerance Analysis ===] (if tolerance specified)
   [Bias as % of Tolerance: [value < 10% for a good system]]
   [Precision/Tolerance Ratio (Cg): [value > 1.33 for a good system]]
   [Accuracy/Tolerance Ratio (Cgk): [value > 1.33 for a good system]]
   [Overall Assessment: [Acceptable/Not Acceptable]]

   [=== Operator Analysis ===] (if operator effects included)
   [Operator | Mean | Bias | Std Dev | n]
   [-----------------------------------]
   [Operator 1 | [value] | [bias] | [stddev] | [count]]
   [Operator 2 | [value] | [bias] | [stddev] | [count]]
   [...]
   ```

5. Visualizations:
   - A series of plots should appear in a separate window, including:
     * Histogram with reference value marked
     * Run chart showing measurements over time or sequence
     * Normal probability plot for checking distribution
     * Operator comparison plot (if applicable)

Error Handling:
1. No Data Loaded:
   - If no data is loaded, a prompt should appear asking if you want to load sample data
   - Selecting "Yes" should load the bias_data.csv file
   - Selecting "No" should return to the main window without analysis

2. Data Validation Errors:
   - If selected columns contain missing values, an error message should appear
   - If fewer than 10 measurements are provided, a warning message should appear
   - If reference values are inconsistent when using a reference column, a warning should appear

3. Interpretation Alerts:
   - If p-value < 0.05, the bias is statistically significant
   - If bias exceeds the acceptable percentage of tolerance, a warning should appear

Troubleshooting:
1. Interpretation Guidelines:
   - A good measurement system should have:
     * Bias not statistically significant (p-value > 0.05)
     * Percent bias < 5%
     * Bias as % of tolerance < 10% (if tolerance is specified)
     * Consistent bias across operators (if applicable)
   
2. Common Issues:
   - If bias is statistically significant, equipment likely needs calibration
   - If bias varies significantly between operators, additional training may be needed
   - If bias changes over time (visible in run chart), the measurement system may be unstable

3. Working with Non-Ideal Data:
   - Minimum 10 measurements are recommended for reliable bias analysis
   - If operators show different bias patterns, analyze them separately
   - If reference value uncertainty is high, results should be interpreted with caution

### 3.5 Stability Study
Test Steps:
1. Navigation:
   - Go to: Quality > Measurement System Analysis > Stability Study

2. Data Input:
   a. Using Sample Data:
   - When prompted "No data is currently loaded. Do you want to load the sample stability data?", click "Yes"
   
   b. Using Existing Data:
   - If you already have data loaded, select:
     * Measurement Column: column containing measurements
     * Time/Date Column: column containing time or date information
     * Operator Column (optional): column identifying who made each measurement
     * Reference Value (optional): known standard value for comparison

3. Dialog Options:
   a. Options Tab:
   - Time Unit Selection: hour, day, week, month (default: hour)
   - Confidence Level: Select "95%" (default)
   - Include operator effects: checked if operator column is available
   - Reference Option:
     * No reference value: selected by default
     * Use reference value: enter a known standard value
   - Trend Detection: checked
   
   b. Graphs Tab:
   - Measurement over Time: checked
   - Control Chart: checked
   - Run Chart: checked
   - Operator Comparison (if applicable): checked
   - Histogram by Time Period: checked
   - Trend Analysis: checked

4. Expected Results:
   ```
   === Stability Study Results ===

   Study Date: [current date and time]
   Data Source: [file name or "New Data"]

   Study Parameters:
     - Measurement Column: Measurement
     - Time/Date Column: Time
     - Time Unit: [hour/day/week/month]
     - Total Measurements: [30-60]
     - Periods/Groups: [number of time periods]
     - Operators: [2-3] (if applicable)
     - Confidence Level: 0.95
     - Reference Value: [value if specified]

   === Overall Statistics ===
   Mean: [value]
   Standard Deviation: [value]
   Upper Control Limit (UCL): [value]
   Lower Control Limit (LCL): [value]
   Violations of Control Limits: [count]

   === Period Statistics ===
   Time Period | Mean | Std Dev | n | Status
   --------------------------------------------
   [Period 1]  | [value] | [value] | [count] | [In Control/Out of Control]
   [Period 2]  | [value] | [value] | [count] | [In Control/Out of Control]
   ...
   
   === Trend Analysis ===
   Slope: [value]
   R-squared: [value]
   p-value: [value]
   Trend Detected: [Yes/No]

   [=== Operator Analysis ===] (if operator included)
   [Operator | Average | Std Dev | Min | Max]
   [----------------------------------------]
   [Op A]    | [value] | [value] | [value] | [value]
   [Op B]    | [value] | [value] | [value] | [value]
   ...

   [=== Reference Comparison ===] (if reference specified)
   [Reference Value: [value]]
   [Average Bias: [value]]
   [Percent of Measurements Within Tolerance: [percentage]]
   ```

5. Report Options:
   - The session window should show a full summary of all statistics
   - The requested plots should be generated in separate windows
   - "Export Report" button should generate a text file containing all statistics

Expected Visualizations:
1. Measurement over Time:
   - Line or scatter plot showing measurements vs time
   - Different colors/symbols for different operators (if applicable)
   - Reference value shown as horizontal line (if specified)

2. Control Chart:
   - Individuals control chart for measurements
   - Center line at overall mean
   - UCL and LCL shown as horizontal lines
   - Out-of-control points highlighted

3. Run Chart:
   - Similar to control chart but without control limits
   - Shows patterns in the data over time
   - Trend line if trend detection enabled

4. Operator Comparison:
   - Box plots or similar showing distribution by operator
   - Statistical comparison of means and variability
   - ANOVA results if appropriate

5. Histogram by Time Period:
   - Distribution of measurements for each time period
   - Reference value marked (if specified)
   - Normal distribution overlay if appropriate

Error Handling:
1. Data Validation Errors:
   - If time/date column is not in proper format, an error message should explain how to format it
   - If measurement column has non-numeric values, an error message should be displayed
   - If fewer than 2 time periods are available, a warning should indicate insufficient data

2. Interpretation Alerts:
   - If trend is detected (p-value < 0.05), warning about unstable measurement system
   - If many control limit violations, alert about potential measurement system issues
   - If large differences between operators, suggest operator training

Troubleshooting:
1. Visualization Issues:
   - If plots appear blank or incorrectly sized, try resizing the dialog window
   - Close and reopen the analysis dialog if visualizations don't display properly

2. Study Assessment:
   - A stable measurement system should show:
     * No significant trend over time (p-value > 0.05)
     * Few or no control limit violations
     * Consistent operator performance (if applicable)
     * Small bias compared to reference value (if specified)

3. Design Requirements:
   - For valid results, the study should have at least:
     * 20 measurements minimum (more is better)
     * At least 5 time periods
     * If using operators, at least 2 operators with multiple measurements each
   - The measurements should be taken of the same part or standard over time

4. Working with Non-Ideal Data:
   - If your data spans different time scales, select the appropriate time unit
   - If reference value is unknown, trends and variability can still be analyzed
   - For unequal sample sizes across periods, the analysis can proceed but interpret results with caution
   - Consider data transformation if measurements are highly skewed

## 4. Six Sigma Tools

### 4.1 DPMO Calculator
Test Steps:
1. Navigation:
   - Go to: Six Sigma > Six Sigma Metrics > DPMO Calculator
2. Data Entry:
   - Enter Process Name: Product
   - Enter Number of Defects: 25
   - Enter Number of Units: 500
   - Enter Number of Opportunities per Unit: 5
   - Click "Calculate"

Expected Results:
- Session Window Output:
  ```
  DPMO Calculation Results for Product
  -----------------------
   NTotal Defects Found: 25
   Defects per Opportunity (DPO): 0.010000
   Defects Per Million Opportunities (DPMO): 10000.00
   Sigma Level: 3.83
   Process Yield: 99.00%
  ```

Error Handling:
- All fields must be positive numbers
- Error if any field is left empty
- Warning for unrealistically high sigma levels

### 4.2 Fishbone Diagram
Test Steps:
1. Navigation:
   - Go to: Six Sigma > DMAIC Tools > Fishbone Diagram

2. Dialog Options:
   a. Problem Statement:
      - Enter a problem statement in the text field (e.g., "High defect rate in production")
   
   b. Categories and Causes:
      - The dialog should show 6 main categories (6M):
        * Materials
        * Methods
        * Machines
        * Manpower (People)
        * Measurement
        * Environment (Mother Nature)
      - For each category, enter at least 2-3 possible causes:
        * Materials: "Poor quality raw materials", "Inconsistent material specifications"
        * Methods: "Outdated procedures", "Lack of standardization"
        * Machines: "Equipment breakdown", "Inadequate maintenance"
        * Manpower: "Insufficient training", "High turnover"
        * Measurement: "Inaccurate gauges", "Subjective quality criteria"
        * Environment: "Poor lighting", "Temperature fluctuations"
   
   c. Diagram Creation:
      - Click "Create Diagram" button

3. Expected Results:
   a. Visualization:
      - A fishbone (Ishikawa) diagram should appear in a new window
      - The diagram should have:
        * A main arrow pointing to the problem statement on the right
        * Six branches (one for each category) connected to the main arrow
        * Each branch should be labeled with its category name
        * Individual causes should be connected to their respective category branches
        * Diagram should be clearly readable and well-formatted
   
   b. Session Window Output:
      ```
      Fishbone Diagram Analysis

      Problem Statement: High defect rate in production

      Categories and Causes:

      Materials:
        - Poor quality raw materials
        - Inconsistent material specifications

      Methods:
        - Outdated procedures
        - Lack of standardization

      Machines:
        - Equipment breakdown
        - Inadequate maintenance

      Manpower:
        - Insufficient training
        - High turnover

      Measurement:
        - Inaccurate gauges
        - Subjective quality criteria

      Environment:
        - Poor lighting
        - Temperature fluctuations
      ```

4. Diagram Customization:
   - Verify that the diagram adjusts layout appropriately when different numbers of causes are entered
   - Categories with more causes should have adequate space
   - Text should not overlap
   - Diagram should scale appropriately

Error Handling:
1. Empty Problem Statement:
   - If the problem statement is left empty, a warning should appear when clicking "Create Diagram"
   - The warning should say "Please enter a problem statement"
   - The diagram should not be created until a problem statement is provided

2. No Causes Entered:
   - If no causes are entered for any category, the diagram should still be created
   - Empty categories should still be displayed but without cause branches
   - An information message may appear: "Some categories have no causes entered"

3. Long Text Entries:
   - Very long problem statements or cause descriptions should be truncated or wrapped in the diagram
   - The full text should still be visible in the session window report

Troubleshooting:
1. Diagram Display:
   - If the diagram appears too small or text is difficult to read, resize the window
   - If causes are not visible, ensure they were entered correctly in the fields
   - If categories appear to overlap, try reducing the number of causes

2. Exporting or Saving:
   - Right-clicking on the diagram should allow saving as an image
   - The session window text can be selected and copied to other applications

3. Best Practices:
   - For clearest results, keep cause descriptions brief and specific
   - Focus on potential root causes rather than symptoms
   - Use action-oriented descriptions where possible
   - Enter causes in order of likelihood or impact for better visualization

### 4.3 FMEA Template
Test Steps:
1. Navigation:
   - Go to: Six Sigma > DMAIC Tools > FMEA Template

2. Project Information:
   - Enter Project Name: "Assembly Process Improvement"
   - Enter Process/Product Name: "PCB Assembly Line"
   - Enter Team Members: "Quality Team, Engineering, Production"
   - Enter Prepared By: "Test User"
   - Select current date using the date picker
   - Enter Revision: "1" (default)

3. FMEA Analysis:
   - The FMEA table should have the following columns:
     * Process Step/Item
     * Potential Failure Mode
     * Potential Effects
     * Severity (1-10)
     * Potential Causes
     * Occurrence (1-10)
     * Current Controls
     * Detection (1-10)
     * RPN
     * Recommended Actions
     * Action Taken
   
   - Enter the following data for three rows:
     * Row 1:
       - Process Step: "Component Placement"
       - Failure Mode: "Missing Component"
       - Effects: "Circuit Failure"
       - Severity: 8
       - Causes: "Pick and Place Machine Error"
       - Occurrence: 3
       - Controls: "AOI Inspection"
       - Detection: 4
       - Recommended Actions: "Implement double-check system"
       - Action Taken: "Added secondary camera inspection"
     
     * Row 2:
       - Process Step: "Wave Soldering"
       - Failure Mode: "Insufficient Solder"
       - Effects: "Weak Connection"
       - Severity: 7
       - Causes: "Incorrect Temperature Setting"
       - Occurrence: 4
       - Controls: "Visual Inspection"
       - Detection: 5
       - Recommended Actions: "Automated temperature monitoring"
       - Action Taken: ""
     
     * Row 3:
       - Process Step: "Final Testing"
       - Failure Mode: "Missed Defect"
       - Effects: "Customer Complaint"
       - Severity: 9
       - Causes: "Incomplete Test Procedure"
       - Occurrence: 2
       - Controls: "Test Protocol Review"
       - Detection: 3
       - Recommended Actions: "Revise test protocol"
       - Action Taken: ""

   - Verify RPN Calculation:
     * For Row 1: Verify that RPN = 8 × 3 × 4 = 96
     * For Row 2: Verify that RPN = 7 × 4 × 5 = 140 (should be highlighted as critical)
     * For Row 3: Verify that RPN = 9 × 2 × 3 = 54

4. Save and Export Options:
   - Click "Save FMEA" button
   - When the save dialog appears, enter a filename and click Save
   - Verify a confirmation message appears
   - Click "Export Report" button
   - When the export dialog appears, enter a filename and click Save
   - Verify a confirmation message appears
   - Verify the report appears in the session window

5. Load FMEA:
   - Click "New FMEA" button to clear data
   - Confirm the clear operation when prompted
   - Click "Load FMEA" button
   - Navigate to and select the previously saved FMEA file
   - Verify all data is loaded correctly

6. Sample Data:
   - Click "New FMEA" button to clear data
   - Click "Load Sample Data" button
   - Verify sample data is loaded into the table
   - Verify RPN values are calculated correctly

Expected Results:
1. Form Elements:
   - A well-organized form with project information section at the top
   - An FMEA table with 11 columns as specified
   - Scoring guidelines section showing interpretation of S, O, D ratings
   - Functional buttons for Save, Load, Export, New, and Close

2. RPN Calculation:
   - RPN should automatically calculate when Severity, Occurrence, or Detection is entered
   - RPN formula should be: Severity × Occurrence × Detection
   - RPN values > 100 should be highlighted in red to indicate critical items

3. File Operations:
   - Save operation should create a JSON file with all FMEA data
   - Load operation should restore all data from a saved file
   - Export operation should create a formatted text report and display it in the session window

Error Handling:
1. Input Validation:
   - If Severity, Occurrence, or Detection values are outside the range 1-10, a warning should appear
   - If invalid numeric values are entered, the RPN should not calculate and an error message may appear

2. File Operations:
   - If a file cannot be saved, loaded, or exported, an appropriate error message should appear
   - If a file format is incorrect during loading, an error message should indicate the issue

3. Empty Fields:
   - Empty cells in the table should be handled gracefully
   - RPN calculation should require all three inputs (S, O, D) to be present

Troubleshooting:
1. If RPN is not calculating:
   - Ensure all three values (Severity, Occurrence, Detection) are entered
   - Ensure values are numerical and within range 1-10
   - Try changing values to trigger recalculation

2. If file operations fail:
   - Check file permissions and disk space
   - Try a different filename or location
   - For loading, ensure the file is a valid FMEA JSON file

3. Best Practices:
   - Process steps should be listed in sequence order
   - Severity ratings should reflect customer impact
   - Occurrence ratings should be based on process history or statistics
   - Detection ratings should reflect control effectiveness
   - Focus on highest RPN items for recommended actions

### 4.4 Sigma Level Calculator
Test Steps:
1. Navigation:
   - Go to: Six Sigma > Six Sigma Metrics > Sigma Level Calculator

2. DPMO to Sigma Calculation:
   - Select "DPMO to Sigma Level" (default selected)
   - Enter DPMO value: 3.4
   - Click "Calculate"

   Expected Results:
   ```
   Sigma Level Calculator Results
   -----------------------------

   Input DPMO: 3.4
   Calculated Sigma Level: 6.00

   Interpretation: World-class performance (Six Sigma)
   ```

3. Sigma to DPMO Calculation:
   - Select "Sigma Level to DPMO"
   - Enter Sigma value: 4.00
   - Click "Calculate"

   Expected Results:
   ```
   Sigma Level Calculator Results
   -----------------------------

   Input Sigma Level: 4.00
   Calculated DPMO: 6210.0

   Interpretation: Industry average
   ```

4. Test Key Values:
   | Input DPMO | Expected Sigma Level | Interpretation |
   |------------|---------------------|----------------|
   | 3.4        | 6.00                | World-class performance (Six Sigma) |
   | 233        | 5.00                | Excellent performance |
   | 6,210      | 4.00                | Industry average |
   | 66,807     | 3.01                | Average performance |

   | Input Sigma | Expected DPMO | Interpretation |
   |-------------|--------------|----------------|
   | 6.00        | 3.4          | World-class performance (Six Sigma) |
   | 5.00        | 233.0        | Excellent performance |
   | 4.00        | 6,210.0      | Industry average |
   | 3.00        | 66,807.0     | Average performance |

Error Handling:
- Negative DPMO values should show an error
- Negative Sigma values should show an error
- Empty input fields should default to 0 for DPMO or 6 for Sigma

Tips and Troubleshooting:
- The calculator includes an informative text area with sigma level interpretations
- The calculation direction determines which input field is enabled
- The calculator implements the standard 1.5 sigma shift formula

### 4.5 Process Yield Analysis
Test Steps:
1. Navigation:
   - Go to: Six Sigma > Six Sigma Metrics > Process Yield Analysis

2. Data Input Methods:
   a. Using Sample Data:
   - When prompted "No data is currently loaded. Do you want to load the sample yield data?", click "Yes"
   
   b. Manual Data Entry (if no sample data available):
   - When prompted, enter:
     * Input Units: 1000
     * Output Units: 950
     * Rework Units: 30
     * Scrap Units: 20
     * Click "OK"
   
   c. Using Existing Data:
   - If you already have data loaded with columns: Input, Output, Rework, Scrap
   - The analysis will use all rows of data in the table

3. Expected Results (for sample data, first row):
   ```
   Process Yield Analysis Results
   ----------------------------

   Input: 1000 units
   Output: 950 units
   Rework: 30 units
   Scrap: 20 units
   
   Calculations:
   First Pass Yield = 92.0%    # (Output - Rework) / Input
   Final Yield = 95.0%         # Output / Input
   Scrap Rate = 2.0%          # Scrap / Input
   Rework Rate = 3.0%         # Rework / Input
   ```

4. Expected Results (for multiple rows of data):
   ```
   Process Yield Analysis - Multiple Batches
   =====================================
   
   Row Input     Output    Rework    Scrap     FPY %     Final %   Scrap %   Rework %  
   --------------------------------------------------------------------------------
   1   1000      950       30        20        92.0%     95.0%     2.0%      3.0%      
   2   1000      960       25        15        93.5%     96.0%     1.5%      2.5%      
   3   1000      970       20        10        95.0%     97.0%     1.0%      2.0%      
   ...
   
   Summary Statistics:
   Average First Pass Yield: 94.7%
   Average Final Yield: 96.5%
   Average Scrap Rate: 1.1%
   Average Rework Rate: 2.2%
   ```

5. Verification Steps:
   - Verify First Pass Yield calculation: (Output - Rework) / Input
   - Verify Final Yield calculation: Output / Input
   - Verify Scrap Rate calculation: Scrap / Input
   - Verify Rework Rate calculation: Rework / Input
   - For multiple rows, verify the summary statistics are correctly calculated

6. Error Handling:
   - Output > Input (should show error: "Output cannot exceed Input")
   - Negative values (should show error: "Output, Rework, and Scrap cannot be negative")
   - Rework + Scrap > Input (should show error: "Rework + Scrap cannot exceed Input")
   - Input ≤ 0 (should show error: "Input units must be positive")
   - Missing required columns (should show warning listing the missing columns)

7. Tips and Troubleshooting:
   - The sample data file is located at sample_data/yield_data.csv
   - For batch-by-batch analysis, load a file with multiple rows
   - The process may provide improvement suggestions if yields are below certain thresholds
   - If no data is available, the tool offers manual data entry through a dialog

### 4.6 Poisson Distribution

#### Overview
This test guide section covers the testing of the Poisson distribution random data generation feature. The Poisson distribution is used to model discrete count data representing the number of events occurring in a fixed interval of time or space.

#### Prerequisites
- The Minitab-like application is installed and running.
- No specific data file is required as this feature generates new data.

#### Test Steps

##### Basic Functionality
1. From the menu bar, select "Calc" → "Random Data" → "Poisson".
2. In the dialog that appears, enter the following values:
   - Sample Size: 100
   - Lambda (average rate): 5
   - Column Name: "Poisson_Test"
   - Keep "Show Distribution Plot" checked
   - Keep "Show Detailed Statistics" checked
3. Click "OK" to generate the data.
4. Verify that the data is generated and displayed in the data table with the specified column name.
5. Verify that the session window displays the following information:
   - Generation parameters (sample size, lambda, column name)
   - Theoretical properties (mean, variance, standard deviation)
   - Probability examples
   - Sample statistics
   - 95% confidence interval
   - Frequency of values
6. Verify that a histogram plot is displayed showing both the generated data and the theoretical probability mass function.

##### Different Parameter Values
1. From the menu bar, select "Calc" → "Random Data" → "Poisson".
2. Set the following values:
   - Sample Size: 500
   - Lambda (average rate): 0.5 (a small value)
   - Column Name: "Poisson_Small"
3. Click "OK" and verify the data and output.
4. Repeat the process with Lambda = 20 (a large value) and Column Name = "Poisson_Large".
5. Verify that the distributions visually match the expected Poisson distribution shape for their respective lambda values:
   - Small lambda (0.5): Should be right-skewed with peak at 0
   - Large lambda (20): Should be approximately symmetric and bell-shaped

##### Column Name Handling
1. Generate a Poisson dataset with column name "PoissonCol".
2. Generate another Poisson dataset with the same column name "PoissonCol".
3. Verify that the second dataset is added with a modified name like "PoissonCol_1" to avoid duplicates.

##### Visualization Options
1. From the menu bar, select "Calc" → "Random Data" → "Poisson".
2. Set Sample Size = 200, Lambda = 10.
3. Uncheck "Show Distribution Plot".
4. Click "OK" and verify that data is generated but no plot is displayed.
5. Repeat with "Show Distribution Plot" checked but "Show Detailed Statistics" unchecked.
6. Verify that the plot appears but the session window only shows basic information, not the detailed statistics.

#### Expected Results
- The Poisson random data should be generated according to the specified parameters.
- The column of data should appear in the table with the specified name (or a modified name if a duplicate).
- The session window should display appropriate statistics and information about the generated data.
- The visualization should correctly display the generated data histogram alongside the theoretical Poisson probability mass function.
- The data values should all be non-negative integers.
- Sample statistics (mean, variance) should be reasonably close to the theoretical values for large samples.

#### Error Handling
1. Try to enter a negative or zero value for Lambda and verify that an error message is displayed.
2. Try to enter a zero value for Sample Size and verify that an error message is displayed.
3. Try to enter a very large value for Sample Size (e.g., 1,000,000) and verify that the application handles it appropriately (either by showing a warning or by performing the generation but potentially taking longer).
4. Try to generate data with an empty Column Name and verify that the default name "Poisson" is used.

#### Troubleshooting
- If the visualization doesn't appear, check if matplotlib is properly installed and configured.
- If the data generation is slow for large sample sizes, this is expected behavior. Consider using a smaller sample size for testing purposes.
- If the data doesn't match the expected Poisson distribution, verify that the lambda parameter is correctly set and that the random number generator is functioning properly.

## 5. File Operations

### 5.1 Open File
Test Steps:
1. Navigation:
   - Go to: File > Open
2. Actions:
   - Select sample_data.csv file
   - Click "Open"

Expected Results:
- File loads successfully
- Data displays in table
- Session window confirms file opened

### 5.2 Save File
Test Steps:
1. Prerequisite:
   - Load sample_data.csv
   - Make some edits to the data
2. Navigation:
   - Go to: File > Save
3. Actions:
   - Confirm save

Expected Results:
- File saves successfully
- Session window confirms save
- Changes persist when reopening the file

### 5.3 Save As
Test Steps:
1. Prerequisite:
   - Load sample_data.csv
2. Navigation:
   - Go to: File > Save As
3. Actions:
   - Enter a new filename
   - Click "Save"

Expected Results:
- New file created
- Session window confirms save as
- Original file remains unchanged

### 5.4 Exit
Test Steps:
1. Navigation:
   - Go to: File > Exit
2. Actions:
   - If unsaved changes, confirm prompt should appear
   - Click "Yes" to save or "No" to discard changes

Expected Results:
- Application closes
- Changes saved or discarded as selected

## 6. Random Data Generation

### 6.1 Normal Distribution
Test Steps:
1. Navigation:
   - Go to: Calc > Random Data > Normal
2. Dialog Options:
   - Set Sample Size: 100 (default)
   - Set Mean: 0 (default)
   - Set Standard Deviation: 1 (default)
   - Check "Create new column in current data" (if you have data loaded)
   - Column Name: "NormalData" (default)
3. Click "OK"

Expected Results:
- If existing data is loaded and "Create new column" is checked:
  - A new column named "NormalData" is added to the table
  - If the name already exists, it will be automatically renamed (e.g., "NormalData_1")
- If no data is loaded or "Create new column" is unchecked:
  - A new dataset is created with a single column of normal random data
- Session Window Output:
  ```
  Normal Random Data Generation
  Sample Size: 100
  Parameters: Mean = 0, Std Dev = 1

  Actual Statistics:
    Mean: [value close to 0]
    Std Dev: [value close to 1]
    Min: [minimum value]
    Max: [maximum value]
  ```
- The generated data should approximate a normal distribution

Error Handling:
- If invalid inputs are provided (e.g., negative standard deviation), appropriate error message is shown
- If sample size is extremely large, application should handle memory constraints appropriately

### 6.2 Uniform Distribution
Test Steps:
1. Navigation:
   - Go to: Calc > Random Data > Uniform
2. Dialog Options:
   - Set Sample Size: 100 (default)
   - Set Minimum Value: 0 (default)
   - Set Maximum Value: 1 (default)
   - Check "Create new column in current data" (if you have data loaded)
   - Column Name: "UniformData" (default)
3. Click "OK"

Expected Results:
- If existing data is loaded and "Create new column" is checked:
  - A new column named "UniformData" is added to the table
  - If the name already exists, it will be automatically renamed
- If no data is loaded or "Create new column" is unchecked:
  - A new dataset is created with a single column of uniform random data
- Session Window Output:
  ```
  Uniform Random Data Generation
  Sample Size: 100
  Parameters: Min = 0, Max = 1

  Actual Statistics:
    Mean: [value close to 0.5]
    Std Dev: [value close to 0.289]
    Min: [minimum value]
    Max: [maximum value]
  ```
- The generated data should be approximately uniformly distributed between the min and max values

Error Handling:
- If maximum value is less than or equal to minimum value, appropriate error message is shown
- If invalid inputs are provided, appropriate error message is shown

### 6.3 Binomial Distribution
Test Steps:
1. Navigation:
   - Go to: Calc > Random Data > Binomial
2. Dialog Options:
   - Set Sample Size: 100 (default)
   - Set Number of Trials: 10 (default)
   - Set Probability of Success: 0.5 (default)
   - Check "Create new column in current data" (if you have data loaded)
   - Column Name: "BinomialData" (default)
3. Click "OK"

Expected Results:
- If existing data is loaded and "Create new column" is checked:
  - A new column named "BinomialData" is added to the table
  - If the name already exists, it will be automatically renamed
- If no data is loaded or "Create new column" is unchecked:
  - A new dataset is created with a single column of binomial random data
- Session Window Output:
  ```
  Binomial Random Data Generation
  Sample Size: 100
  Parameters: Trials = 10, Probability = 0.5

  Actual Statistics:
    Mean: [value close to 5]
    Std Dev: [value close to 1.58]
    Min: [minimum value]
    Max: [maximum value]
  ```
- The generated data should consist of integer values between 0 and the number of trials

Error Handling:
- If probability is not between 0 and 1, appropriate error message is shown
- If number of trials is negative, appropriate error message is shown

### 6.4 Poisson Distribution
Test Steps:
1. Navigation:
   - Go to: Calc > Random Data > Poisson
2. Dialog Options:
   - Set Sample Size: 100 (default)
   - Set Lambda (mean): 5 (default)
   - Check "Create new column in current data" (if you have data loaded)
   - Column Name: "PoissonData" (default)
3. Click "OK"

Expected Results:
- If existing data is loaded and "Create new column" is checked:
  - A new column named "PoissonData" is added to the table
  - If the name already exists, it will be automatically renamed
- If no data is loaded or "Create new column" is unchecked:
  - A new dataset is created with a single column of Poisson random data
- Session Window Output:
  ```
  Poisson Random Data Generation
  Sample Size: 100
  Parameters: Lambda = 5

  Actual Statistics:
    Mean: [value close to 5]
    Std Dev: [value close to 2.24]
    Min: [minimum value]
    Max: [maximum value]
  ```
- The generated data should consist of non-negative integer values

Error Handling:
- If lambda (mean) is negative, appropriate error message is shown
- If extremely large lambda value is entered, application should handle computations appropriately

### General Features for Random Data Generation
Common Test Steps (applicable to all distributions):
1. When a data table already exists:
   - Load sample_data.csv
   - Generate random data with "Create new column in current data" checked
   - Verify that the new column is added to the existing data

2. When no data table exists:
   - Close any open file without saving
   - Generate random data with "Create new column in current data" unchecked
   - Verify that a new data table is created with just the random data column

3. Column naming conflicts:
   - Generate random data with a column name that already exists
   - Verify that the application adds a suffix to ensure uniqueness (e.g., "NormalData_1")

4. Data size handling:
   - Load a dataset with 50 rows
   - Generate 100 random data points with "Create new column in current data" checked
   - Verify that the table expands to accommodate all 100 data points
   - Generate 25 random data points with "Create new column in current data" checked
   - Verify that the first 25 rows contain data and the rest of the rows in that column contain NaN

## 7. Advanced Testing Considerations

### Cross-Feature Integration
Test workflows that combine multiple features, such as:
1. Load data
2. Perform descriptive statistics
3. Create a hypothesis test
4. Generate a visualization
5. Save the dataset with results

### Stress Testing
1. Test with large datasets (1000+ rows)
2. Test with many columns (50+)
3. Test with various data types and missing values

### Error Recovery
Verify the application can recover from:
1. Invalid user inputs
2. File access issues
3. Memory constraints
4. Calculation errors

### Interoperability
Test importing/exporting with:
1. Different CSV formats (various delimiters)
2. Excel files (.xlsx, .xls)
3. Other statistical software formats