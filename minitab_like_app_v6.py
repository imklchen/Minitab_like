import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
from statsmodels.formula.api import ols
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget, QMenuBar, QMenu, QTextEdit, QFileDialog, 
                             QMessageBox, QInputDialog, QDialog, QFormLayout, QLineEdit,
                             QPushButton, QLabel, QComboBox, QVBoxLayout, QHBoxLayout, QGroupBox)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
import seaborn as sns

def calculate_control_limits(data, n=1):
    """Calculate control limits for X-bar and R charts"""
    xbar = np.mean(data)
    r = np.max(data) - np.min(data)
    
    # Constants for control limits
    A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577}
    D3 = {2: 0, 3: 0, 4: 0, 5: 0}
    D4 = {2: 3.267, 3: 2.575, 4: 2.282, 5: 2.115}
    
    # Use n=2 if not in constants dictionary
    n = n if n in A2 else 2
    
    # Calculate control limits
    ucl_x = xbar + A2[n] * r
    lcl_x = xbar - A2[n] * r
    ucl_r = D4[n] * r
    lcl_r = D3[n] * r
    
    return {
        'center_x': xbar,
        'ucl_x': ucl_x,
        'lcl_x': lcl_x,
        'center_r': r,
        'ucl_r': ucl_r,
        'lcl_r': lcl_r
    }

def calculate_capability_indices(data, usl, lsl):
    """Calculate process capability indices"""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    # Calculate Cp
    cp = (usl - lsl) / (6 * std)
    
    # Calculate Cpu and Cpl
    cpu = (usl - mean) / (3 * std)
    cpl = (mean - lsl) / (3 * std)
    
    # Calculate Cpk
    cpk = min(cpu, cpl)
    
    return {
        'cp': cp,
        'cpu': cpu,
        'cpl': cpl,
        'cpk': cpk
    }

def calculate_dpmo(defects, opportunities, units):
    """Calculate Defects Per Million Opportunities"""
    return (defects / (opportunities * units)) * 1000000

def dpmo_to_sigma(dpmo):
    """Convert DPMO to Sigma Level"""
    return 0.8406 + np.sqrt(29.37 - 2.221 * np.log(dpmo))

def create_pareto_chart(categories, values):
    """Create a Pareto chart"""
    df = pd.DataFrame({'categories': categories, 'values': values})
    df = df.sort_values('values', ascending=False)
    
    cumulative_percentage = np.cumsum(df['values']) / sum(df['values']) * 100
    
    fig, ax1 = plt.subplots()
    
    ax1.bar(df['categories'], df['values'])
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Frequency')
    
    ax2 = ax1.twinx()
    ax2.plot(df['categories'], cumulative_percentage, 'r-')
    ax2.set_ylabel('Cumulative Percentage')
    
    plt.title('Pareto Chart')
    return fig

class MinitabLikeApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Custom Minitab-Like Tool")
        self.resize(900, 600)
        self.data = pd.DataFrame(columns=[f"C{i+1}" for i in range(10)])
        self.initUI()

    def initUI(self):
        self.table = QTableWidget(50, 10)
        self.table.setHorizontalHeaderLabels(self.data.columns)

        self.sessionWindow = QTextEdit()
        self.sessionWindow.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.sessionWindow)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.createMenuBar()

    def createMenuBar(self):
        menuBar = QMenuBar()
        self.setMenuBar(menuBar)

        fileMenu = QMenu("File", self)
        calcMenu = QMenu("Calc", self)
        statMenu = QMenu("Stat", self)
        qualityMenu = QMenu("Quality", self)
        sixSigmaMenu = QMenu("Six Sigma", self)

        # File Menu
        fileMenu.addAction(self.makeAction("Open", self.openFile))
        fileMenu.addAction(self.makeAction("Save", self.saveFile))
        fileMenu.addAction(self.makeAction("Exit", self.close))

        # Stat Menu
        basicStatMenu = QMenu("Basic Statistics", self)
        statMenu.addMenu(basicStatMenu)
        basicStatMenu.addAction(self.makeAction("Descriptive Statistics", self.calculateDescriptiveStats))
        basicStatMenu.addAction(self.makeAction("Correlation Analysis", self.calculateCorrelation))
        basicStatMenu.addAction(self.makeAction("Probability Analysis", self.probabilityAnalysis))

        advancedStatMenu = QMenu("Advanced Statistics", self)
        statMenu.addMenu(advancedStatMenu)
        advancedStatMenu.addAction(self.makeAction("Hypothesis Testing", self.hypothesisTesting))
        advancedStatMenu.addAction(self.makeAction("ANOVA", self.performANOVA))
        advancedStatMenu.addAction(self.makeAction("Regression Analysis", self.regressionAnalysis))
        advancedStatMenu.addAction(self.makeAction("Chi-Square Tests", self.chiSquareTests))

        # Add Design of Experiments directly to Stat menu
        statMenu.addAction(self.makeAction("Create DOE", self.createDOE))
        statMenu.addAction(self.makeAction("Analyze DOE", self.analyzeDOE))

        # Quality Menu
        controlChartsMenu = QMenu("Control Charts", self)
        qualityMenu.addMenu(controlChartsMenu)
        controlChartsMenu.addAction(self.makeAction("X-bar R Chart", self.xbarRChart))
        controlChartsMenu.addAction(self.makeAction("Individual Chart", self.individualChart))
        controlChartsMenu.addAction(self.makeAction("Moving Range Chart", self.movingRangeChart))

        capabilityMenu = QMenu("Capability Analysis", self)
        qualityMenu.addMenu(capabilityMenu)
        capabilityMenu.addAction(self.makeAction("Process Capability", self.processCapability))
        capabilityMenu.addAction(self.makeAction("Measurement System Analysis", self.msaAnalysis))

        # Six Sigma Menu
        dmaic = QMenu("DMAIC Tools", self)
        sixSigmaMenu.addMenu(dmaic)
        dmaic.addAction(self.makeAction("Pareto Chart", self.paretoChart))
        dmaic.addAction(self.makeAction("Fishbone Diagram", self.fishboneDiagram))
        dmaic.addAction(self.makeAction("FMEA Template", self.fmeaTemplate))

        metrics = QMenu("Six Sigma Metrics", self)
        sixSigmaMenu.addMenu(metrics)
        metrics.addAction(self.makeAction("DPMO Calculator", self.dpmoCalculator))
        metrics.addAction(self.makeAction("Sigma Level Calculator", self.sigmaLevelCalc))
        metrics.addAction(self.makeAction("Process Yield Analysis", self.yieldAnalysis))

        # Calc Menu
        randomMenu = QMenu("Random Data", self)
        calcMenu.addMenu(randomMenu)
        randomMenu.addAction(self.makeAction("Normal", lambda: self.generateRandomData("Normal")))
        randomMenu.addAction(self.makeAction("Binomial", lambda: self.generateRandomData("Binomial")))
        randomMenu.addAction(self.makeAction("Uniform", lambda: self.generateRandomData("Uniform")))

        probDistMenu = QMenu("Probability Distributions", self)
        calcMenu.addMenu(probDistMenu)
        probDistMenu.addAction(self.makeAction("Poisson", self.poissonDistribution))

        # Add all menus to menubar
        menuBar.addMenu(fileMenu)
        menuBar.addMenu(statMenu)
        menuBar.addMenu(qualityMenu)
        menuBar.addMenu(sixSigmaMenu)
        menuBar.addMenu(calcMenu)

    def makeAction(self, name, func):
        action = QAction(name, self)
        action.triggered.connect(func)
        return action

    def loadDataFromTable(self):
        rows, cols = self.table.rowCount(), self.table.columnCount()
        data = {}
        for col in range(cols):
            column_data = []
            for row in range(rows):
                item = self.table.item(row, col)
                if item and item.text().strip():
                    text = item.text().strip()
                    try:
                        column_data.append(float(text))
                    except ValueError:
                        column_data.append(text)
                else:
                    column_data.append(np.nan)
            col_name = self.table.horizontalHeaderItem(col).text()
            data[col_name] = column_data
        self.data = pd.DataFrame(data)

    def openFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Excel Files (*.xlsx);;CSV Files (*.csv)")
        if filename:
            if filename.endswith('.csv'):
                self.data = pd.read_csv(filename)
            else:
                self.data = pd.read_excel(filename)
            self.updateTable()

    def saveFile(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if filename:
            self.loadDataFromTable()
            if filename.endswith('.csv'):
                self.data.to_csv(filename, index=False)
            else:
                self.data.to_excel(filename, index=False)

    def updateTable(self):
        self.table.setRowCount(len(self.data))
        self.table.setColumnCount(len(self.data.columns))
        self.table.setHorizontalHeaderLabels(self.data.columns)
        for row in range(len(self.data)):
            for col in range(len(self.data.columns)):
                value = self.data.iat[row, col]
                if pd.notna(value):
                    self.table.setItem(row, col, QTableWidgetItem(str(value)))

    def calculateDescriptiveStats(self):
        self.loadDataFromTable()
        stats_output = self.data.describe().to_string()
        self.sessionWindow.setText(f"Descriptive Statistics:\n\n{stats_output}")

    def calculateCorrelation(self):
        self.loadDataFromTable()
        corr_matrix = self.data.corr(numeric_only=True)
        self.sessionWindow.setText(f"Correlation Matrix:\n\n{corr_matrix}")

    def probabilityAnalysis(self):
        self.loadDataFromTable()
        col = self.selectColumnDialog()
        if col:
            data = pd.to_numeric(self.data[col], errors='coerce').dropna()
            mu, sigma = np.mean(data), np.std(data)
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
            y = stats.norm.pdf(x, mu, sigma)
            plt.plot(x, y, label=f"Normal Dist (μ={mu:.2f}, σ={sigma:.2f})")
            plt.hist(data, bins=10, density=True, alpha=0.5)
            plt.legend()
            plt.show()

    def selectColumnDialog(self):
        col_names = self.data.columns.tolist()
        col, ok = QInputDialog.getItem(self, "Select Column", "Choose column:", col_names, 0, False)
        return col if ok else None

    def generateRandomData(self, dist_type):
        size, ok = QInputDialog.getInt(self, "Size", f"How many random numbers ({dist_type})?", 100, 1, 1000)
        if not ok:
            return
        if dist_type == "Normal":
            data = np.random.normal(0, 1, size)
        elif dist_type == "Binomial":
            data = np.random.binomial(10, 0.5, size)
        elif dist_type == "Uniform":
            data = np.random.uniform(0, 1, size)
        self.data = pd.DataFrame({dist_type: data})
        self.updateTable()

    def poissonDistribution(self):
        calculation_type, ok = QInputDialog.getItem(self, "Poisson Distribution", 
            "Choose Calculation Type:", 
            ["Probability", "Cumulative Probability", "Inverse Cumulative Probability"], 0, False)
        if not ok:
            return

        mean, ok = QInputDialog.getDouble(self, "Poisson Distribution", "Enter Mean (Average number of events):", 3.0, 0.1, 1000, 2)
        if not ok:
            return

        column = self.selectColumnDialog()
        if not column:
            return

        x_values = pd.to_numeric(self.data[column], errors='coerce').dropna().tolist()

        result_text = f"Poisson with mean = {mean}\n\n"

        if calculation_type == "Inverse Cumulative Probability":
            result_text += "P(X ≤ x)\t x\n"
            for prob in x_values:
                x = math.ceil(stats.poisson.ppf(prob, mean))  # FIX: use math.ceil() for discrete Poisson
                result_text += f"{prob:.6f}\t{x}\n"
        else:
            result_text += "x\tP(X ≤ x) / P(X = x)\n"
            for x in x_values:
                if calculation_type == "Probability":
                    prob = stats.poisson.pmf(x, mean)
                else:
                    prob = stats.poisson.cdf(x, mean)
                result_text += f"{int(x)}\t{prob:.6f}\n"

        self.sessionWindow.setText(result_text)

    def xbarRChart(self):
        """Create X-bar and R control charts"""
        col = self.selectColumnDialog()
        if not col:
            return
        
        data = pd.to_numeric(self.data[col], errors='coerce').dropna()
        if len(data) < 2:
            QMessageBox.warning(self, "Warning", "Insufficient data for control charts")
            return
            
        limits = calculate_control_limits(data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # X-bar chart
        ax1.plot(data, marker='o')
        ax1.axhline(y=limits['center_x'], color='g', linestyle='-', label='Center')
        ax1.axhline(y=limits['ucl_x'], color='r', linestyle='--', label='UCL')
        ax1.axhline(y=limits['lcl_x'], color='r', linestyle='--', label='LCL')
        ax1.set_title('X-bar Chart')
        ax1.legend()
        
        # R chart
        ranges = np.array([max(data) - min(data)])
        ax2.plot(ranges, marker='o')
        ax2.axhline(y=limits['center_r'], color='g', linestyle='-', label='Center')
        ax2.axhline(y=limits['ucl_r'], color='r', linestyle='--', label='UCL')
        ax2.axhline(y=limits['lcl_r'], color='r', linestyle='--', label='LCL')
        ax2.set_title('R Chart')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def processCapability(self):
        """Perform process capability analysis"""
        col = self.selectColumnDialog()
        if not col:
            return
            
        data = pd.to_numeric(self.data[col], errors='coerce').dropna()
        
        # Get specification limits from user
        usl, ok1 = QInputDialog.getDouble(self, "Process Capability", "Enter Upper Specification Limit:", 0)
        if not ok1:
            return
        lsl, ok2 = QInputDialog.getDouble(self, "Process Capability", "Enter Lower Specification Limit:", 0)
        if not ok2:
            return
            
        # Calculate capability indices
        indices = calculate_capability_indices(data, usl, lsl)
        
        # Create capability report
        report = f"""Process Capability Analysis for {col}

Specification Limits:
USL: {usl:.3f}
LSL: {lsl:.3f}

Capability Indices:
Cp: {indices['cp']:.3f}
Cpu: {indices['cpu']:.3f}
Cpl: {indices['cpl']:.3f}
Cpk: {indices['cpk']:.3f}

Process Statistics:
Mean: {np.mean(data):.3f}
StDev: {np.std(data, ddof=1):.3f}
"""
        self.sessionWindow.setText(report)
        
        # Create capability plot
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, density=True, alpha=0.7)
        
        # Add normal curve
        x = np.linspace(min(data), max(data), 100)
        y = stats.norm.pdf(x, np.mean(data), np.std(data, ddof=1))
        plt.plot(x, y, 'r-', lw=2)
        
        # Add specification limits
        plt.axvline(x=usl, color='r', linestyle='--', label='USL')
        plt.axvline(x=lsl, color='r', linestyle='--', label='LSL')
        
        plt.title(f'Process Capability Plot for {col}')
        plt.legend()
        plt.show()

    def paretoChart(self):
        """Create a Pareto chart"""
        # Get categories and values from user
        categories_col = self.selectColumnDialog()
        if not categories_col:
            return
        
        values_col = self.selectColumnDialog()
        if not values_col:
            return
            
        categories = self.data[categories_col].dropna().tolist()
        values = pd.to_numeric(self.data[values_col], errors='coerce').dropna().tolist()
        
        if len(categories) != len(values):
            QMessageBox.warning(self, "Warning", "Categories and values must have the same length")
            return
            
        fig = create_pareto_chart(categories, values)
        plt.show()

    def dpmoCalculator(self):
        """Calculate DPMO and Sigma Level"""
        defects, ok1 = QInputDialog.getInt(self, "DPMO Calculator", "Enter number of defects:", 0, 0)
        if not ok1:
            return
            
        opportunities, ok2 = QInputDialog.getInt(self, "DPMO Calculator", "Enter opportunities per unit:", 1, 1)
        if not ok2:
            return
            
        units, ok3 = QInputDialog.getInt(self, "DPMO Calculator", "Enter number of units:", 1, 1)
        if not ok3:
            return
            
        dpmo = calculate_dpmo(defects, opportunities, units)
        sigma = dpmo_to_sigma(dpmo)
        
        report = f"""DPMO Analysis Results

Number of Defects: {defects}
Opportunities per Unit: {opportunities}
Number of Units: {units}

DPMO: {dpmo:.2f}
Sigma Level: {sigma:.2f}
"""
        self.sessionWindow.setText(report)

    def hypothesisTesting(self):
        """Perform hypothesis testing"""
        # Create dialog for test selection
        test_type, ok = QInputDialog.getItem(self, "Hypothesis Testing",
            "Select Test Type:",
            ["One-Sample t-Test", "Two-Sample t-Test", "Paired t-Test"], 0, False)
        if not ok:
            return

        if test_type == "One-Sample t-Test":
            self.one_sample_t_test()
        elif test_type == "Two-Sample t-Test":
            self.two_sample_t_test()
        elif test_type == "Paired t-Test":
            self.paired_t_test()

    def one_sample_t_test(self):
        """Perform one-sample t-test"""
        # Select data column
        col = self.selectColumnDialog()
        if not col:
            return

        # Get hypothesized mean
        hypothesized_mean, ok = QInputDialog.getDouble(self, "One-Sample t-Test",
            "Enter hypothesized mean:", 0.0)
        if not ok:
            return

        # Get significance level
        alpha, ok = QInputDialog.getDouble(self, "One-Sample t-Test",
            "Enter significance level (α):", 0.05, 0.01, 0.1, 3)
        if not ok:
            return

        # Get alternative hypothesis
        alternative, ok = QInputDialog.getItem(self, "One-Sample t-Test",
            "Select alternative hypothesis:",
            ["two-sided", "less", "greater"], 0, False)
        if not ok:
            return

        # Perform test
        data = pd.to_numeric(self.data[col], errors='coerce').dropna()
        t_stat, p_value = stats.ttest_1samp(data, hypothesized_mean, alternative=alternative)

        # Create report
        report = f"""One-Sample t-Test Results for {col}

Null Hypothesis: μ = {hypothesized_mean:.3f}
Alternative Hypothesis: μ {self._get_alternative_symbol(alternative)} {hypothesized_mean:.3f}
Significance Level (α): {alpha:.3f}

Test Statistics:
t-statistic: {t_stat:.4f}
p-value: {p_value:.4f}

Sample Statistics:
Sample Size: {len(data)}
Sample Mean: {np.mean(data):.3f}
Sample Std Dev: {np.std(data, ddof=1):.3f}

Conclusion: {'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis at α = {alpha:.3f}
"""
        self.sessionWindow.setText(report)

        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=10, density=True, alpha=0.7, label='Sample Data')
        
        # Add hypothesized mean line
        plt.axvline(x=hypothesized_mean, color='r', linestyle='--', label='Hypothesized Mean')
        
        # Add sample mean line
        plt.axvline(x=np.mean(data), color='g', linestyle='--', label='Sample Mean')
        
        plt.title(f'One-Sample t-Test for {col}')
        plt.legend()
        plt.show()

    def two_sample_t_test(self):
        """Perform two-sample t-test"""
        # Select first sample
        col1 = self.selectColumnDialog()
        if not col1:
            return

        # Select second sample
        col2 = self.selectColumnDialog()
        if not col2:
            return

        # Get significance level
        alpha, ok = QInputDialog.getDouble(self, "Two-Sample t-Test",
            "Enter significance level (α):", 0.05, 0.01, 0.1, 3)
        if not ok:
            return

        # Get alternative hypothesis
        alternative, ok = QInputDialog.getItem(self, "Two-Sample t-Test",
            "Select alternative hypothesis:",
            ["two-sided", "less", "greater"], 0, False)
        if not ok:
            return

        # Perform test
        data1 = pd.to_numeric(self.data[col1], errors='coerce').dropna()
        data2 = pd.to_numeric(self.data[col2], errors='coerce').dropna()
        t_stat, p_value = stats.ttest_ind(data1, data2, alternative=alternative)

        # Create report
        report = f"""Two-Sample t-Test Results

Null Hypothesis: μ₁ = μ₂
Alternative Hypothesis: μ₁ {self._get_alternative_symbol(alternative)} μ₂
Significance Level (α): {alpha:.3f}

Test Statistics:
t-statistic: {t_stat:.4f}
p-value: {p_value:.4f}

Sample Statistics:
{col1}:
  Sample Size: {len(data1)}
  Sample Mean: {np.mean(data1):.3f}
  Sample Std Dev: {np.std(data1, ddof=1):.3f}

{col2}:
  Sample Size: {len(data2)}
  Sample Mean: {np.mean(data2):.3f}
  Sample Std Dev: {np.std(data2, ddof=1):.3f}

Conclusion: {'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis at α = {alpha:.3f}
"""
        self.sessionWindow.setText(report)

        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.boxplot([data1, data2], labels=[col1, col2])
        plt.title('Two-Sample t-Test Box Plot')
        plt.ylabel('Values')
        plt.show()

    def paired_t_test(self):
        """Perform paired t-test"""
        # Select before and after columns
        before_col = self.selectColumnDialog()
        if not before_col:
            return

        after_col = self.selectColumnDialog()
        if not after_col:
            return

        # Get significance level
        alpha, ok = QInputDialog.getDouble(self, "Paired t-Test",
            "Enter significance level (α):", 0.05, 0.01, 0.1, 3)
        if not ok:
            return

        # Get alternative hypothesis
        alternative, ok = QInputDialog.getItem(self, "Paired t-Test",
            "Select alternative hypothesis:",
            ["two-sided", "less", "greater"], 0, False)
        if not ok:
            return

        # Perform test
        before_data = pd.to_numeric(self.data[before_col], errors='coerce').dropna()
        after_data = pd.to_numeric(self.data[after_col], errors='coerce').dropna()
        t_stat, p_value = stats.ttest_rel(before_data, after_data, alternative=alternative)

        # Create report
        report = f"""Paired t-Test Results

Null Hypothesis: μ_diff = 0
Alternative Hypothesis: μ_diff {self._get_alternative_symbol(alternative)} 0
Significance Level (α): {alpha:.3f}

Test Statistics:
t-statistic: {t_stat:.4f}
p-value: {p_value:.4f}

Sample Statistics:
Before ({before_col}):
  Sample Size: {len(before_data)}
  Sample Mean: {np.mean(before_data):.3f}
  Sample Std Dev: {np.std(before_data, ddof=1):.3f}

After ({after_col}):
  Sample Size: {len(after_data)}
  Sample Mean: {np.mean(after_data):.3f}
  Sample Std Dev: {np.std(after_data, ddof=1):.3f}

Mean Difference: {np.mean(after_data - before_data):.3f}
Std Dev of Differences: {np.std(after_data - before_data, ddof=1):.3f}

Conclusion: {'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis at α = {alpha:.3f}
"""
        self.sessionWindow.setText(report)

        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(before_data)), before_data, 'b-', label='Before')
        plt.plot(range(len(after_data)), after_data, 'r-', label='After')
        plt.title('Paired t-Test: Before vs After')
        plt.xlabel('Sample')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

    def _get_alternative_symbol(self, alternative):
        """Helper function to get the mathematical symbol for alternative hypothesis"""
        if alternative == "two-sided":
            return "≠"
        elif alternative == "less":
            return "<"
        else:  # greater
            return ">"

    def performANOVA(self):
        """Perform ANOVA analysis"""
        # Create dialog for ANOVA type selection
        anova_type, ok = QInputDialog.getItem(self, "ANOVA Analysis",
            "Select ANOVA Type:",
            ["One-Way ANOVA", "Two-Way ANOVA"], 0, False)
        if not ok:
            return

        if anova_type == "One-Way ANOVA":
            self.one_way_anova()
        else:
            self.two_way_anova()

    def one_way_anova(self):
        """Perform one-way ANOVA"""
        # Select response variable
        response_col = self.selectColumnDialog()
        if not response_col:
            return

        # Select factor variable
        factor_col = self.selectColumnDialog()
        if not factor_col:
            return

        # Get significance level
        alpha, ok = QInputDialog.getDouble(self, "One-Way ANOVA",
            "Enter significance level (α):", 0.05, 0.01, 0.1, 3)
        if not ok:
            return

        # Prepare data for ANOVA
        response_data = pd.to_numeric(self.data[response_col], errors='coerce')
        factor_data = self.data[factor_col]
        
        # Group data by factor levels
        groups = []
        group_labels = []
        for level in factor_data.unique():
            if pd.notna(level):
                group_data = response_data[factor_data == level].dropna()
                if len(group_data) > 0:
                    groups.append(group_data)
                    group_labels.append(str(level))

        if len(groups) < 2:
            QMessageBox.warning(self, "Warning", "Need at least 2 groups for ANOVA")
            return

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Calculate descriptive statistics
        desc_stats = []
        for i, group in enumerate(groups):
            desc_stats.append({
                'Group': group_labels[i],
                'N': len(group),
                'Mean': np.mean(group),
                'Std Dev': np.std(group, ddof=1)
            })

        # Create report
        report = f"""One-Way ANOVA Results

Response Variable: {response_col}
Factor: {factor_col}
Significance Level (α): {alpha:.3f}

ANOVA Table:
Source          DF    SS        MS        F        P
Factor         {len(groups)-1}    {f_stat:.4f}    {f_stat/(len(groups)-1):.4f}    {f_stat:.4f}    {p_value:.4f}
Error          {sum(len(g)-1 for g in groups)}    {sum((len(g)-1)*np.var(g, ddof=1) for g in groups):.4f}    {sum((len(g)-1)*np.var(g, ddof=1) for g in groups)/sum(len(g)-1 for g in groups):.4f}
Total          {sum(len(g)-1 for g in groups) + len(groups)-1}    {f_stat + sum((len(g)-1)*np.var(g, ddof=1) for g in groups):.4f}

Descriptive Statistics:
Group    N    Mean    Std Dev
"""
        for stat in desc_stats:
            report += f"{stat['Group']:<8} {stat['N']:<4} {stat['Mean']:.3f}  {stat['Std Dev']:.3f}\n"

        report += f"\nConclusion: {'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis at α = {alpha:.3f}"
        self.sessionWindow.setText(report)

        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.boxplot(groups, labels=group_labels)
        plt.title(f'One-Way ANOVA: {response_col} by {factor_col}')
        plt.xlabel(factor_col)
        plt.ylabel(response_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def two_way_anova(self):
        """Perform two-way ANOVA"""
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols
        except ImportError:
            QMessageBox.warning(self, "Warning", 
                "statsmodels package is required for Two-Way ANOVA.\n"
                "Please install it using:\n"
                "pip install statsmodels")
            return

        # Select response variable
        response_col = self.selectColumnDialog()
        if not response_col:
            return

        # Select first factor
        factor1_col = self.selectColumnDialog()
        if not factor1_col:
            return

        # Select second factor
        factor2_col = self.selectColumnDialog()
        if not factor2_col:
            return

        # Get significance level
        alpha, ok = QInputDialog.getDouble(self, "Two-Way ANOVA",
            "Enter significance level (α):", 0.05, 0.01, 0.1, 3)
        if not ok:
            return

        # Prepare data for ANOVA
        response_data = pd.to_numeric(self.data[response_col], errors='coerce')
        factor1_data = self.data[factor1_col]
        factor2_data = self.data[factor2_col]

        # Check for missing values
        if response_data.isna().any() or factor1_data.isna().any() or factor2_data.isna().any():
            QMessageBox.warning(self, "Warning", 
                "Data contains missing values. Please ensure all data is complete.")
            return

        # Create DataFrame for analysis
        df = pd.DataFrame({
            response_col: response_data,
            factor1_col: factor1_data,
            factor2_col: factor2_data
        })

        # Check if we have enough unique values in each factor
        if len(df[factor1_col].unique()) < 2 or len(df[factor2_col].unique()) < 2:
            QMessageBox.warning(self, "Warning", 
                "Each factor must have at least 2 unique values.")
            return

        try:
            # Create formula for two-way ANOVA
            formula = f"{response_col} ~ C({factor1_col}) + C({factor2_col}) + C({factor1_col}):C({factor2_col})"
            
            # Fit the model
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Create report
            report = f"""Two-Way ANOVA Results

Response Variable: {response_col}
Factor 1: {factor1_col}
Factor 2: {factor2_col}
Significance Level (α): {alpha:.3f}

ANOVA Table:
{anova_table.to_string()}

Model Summary:
R-squared: {model.rsquared:.4f}
Adjusted R-squared: {model.rsquared_adj:.4f}
F-statistic: {model.fvalue:.4f}
p-value: {model.f_pvalue:.4f}

Descriptive Statistics:
{df.groupby([factor1_col, factor2_col])[response_col].describe().to_string()}

Conclusion: {'Reject' if model.f_pvalue < alpha else 'Fail to reject'} the null hypothesis at α = {alpha:.3f}
"""
            self.sessionWindow.setText(report)

            # Create visualization
            plt.figure(figsize=(12, 6))
            
            # Create interaction plot
            interaction_data = df.groupby([factor1_col, factor2_col])[response_col].mean().unstack()
            interaction_data.plot(kind='bar', width=0.8)
            plt.title(f'Two-Way ANOVA: {response_col} by {factor1_col} and {factor2_col}')
            plt.xlabel(factor1_col)
            plt.ylabel(response_col)
            plt.legend(title=factor2_col)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred during ANOVA analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def regressionAnalysis(self):
        """Perform regression analysis"""
        try:
            from statsmodels.formula.api import ols
            import statsmodels.api as sm
        except ImportError:
            QMessageBox.warning(self, "Warning", 
                "statsmodels package is required for Regression Analysis.\n"
                "Please install it using:\n"
                "pip install statsmodels")
            return

        # Create dialog for regression type selection
        regression_type, ok = QInputDialog.getItem(self, "Regression Analysis",
            "Select Regression Type:",
            ["Simple Linear Regression", "Multiple Linear Regression"], 0, False)
        if not ok:
            return

        if regression_type == "Simple Linear Regression":
            self.simple_linear_regression()
        else:
            self.multiple_linear_regression()

    def simple_linear_regression(self):
        """Perform simple linear regression"""
        # Select dependent variable (Y)
        y_col = self.selectColumnDialog()
        if not y_col:
            return

        # Select independent variable (X)
        x_col = self.selectColumnDialog()
        if not x_col:
            return

        # Prepare data
        try:
            # Create DataFrame with selected columns
            df = pd.DataFrame({
                'y': pd.to_numeric(self.data[y_col], errors='coerce'),
                'x': pd.to_numeric(self.data[x_col], errors='coerce')
            })
            
            # Remove missing values
            df = df.dropna()

            if len(df) < 2:
                QMessageBox.warning(self, "Warning", "Insufficient data for regression analysis")
                return

            # Fit regression model using formula interface
            model = ols('y ~ x', data=df).fit()
            
            # Calculate correlation coefficient
            correlation = df['x'].corr(df['y'])

            # Create regression equation string
            equation = f"Y = {model.params.iloc[0]:.4f} + {model.params.iloc[1]:.4f}X"

            # Create report
            report = f"""Simple Linear Regression Analysis

Dependent Variable (Y): {y_col}
Independent Variable (X): {x_col}

Regression Equation:
{equation}

Model Summary:
R-squared: {model.rsquared:.4f}
Adjusted R-squared: {model.rsquared_adj:.4f}
Correlation Coefficient (r): {correlation:.4f}

Analysis of Variance:
{model.summary().tables[0]}

Coefficients:
{model.summary().tables[1]}
"""
            self.sessionWindow.setText(report)

            # Create visualization
            plt.figure(figsize=(12, 6))
            
            # Scatter plot of data points
            plt.scatter(df['x'], df['y'], color='blue', alpha=0.5, label='Data Points')
            
            # Regression line
            x_range = np.linspace(df['x'].min(), df['x'].max(), 100)
            y_pred = model.params.iloc[0] + model.params.iloc[1] * x_range
            plt.plot(x_range, y_pred, color='red', label='Regression Line')
            
            plt.title(f'Simple Linear Regression\n{y_col} vs {x_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred during regression analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def multiple_linear_regression(self):
        """Perform multiple linear regression"""
        # Select dependent variable (Y)
        y_col = self.selectColumnDialog()
        if not y_col:
            return

        # Select independent variables (X)
        x_cols = []
        while True:
            x_col = self.selectColumnDialog()
            if not x_col:
                break
            if x_col in x_cols:
                QMessageBox.warning(self, "Warning", "Variable already selected")
                continue
            x_cols.append(x_col)
            
            # Ask if user wants to add another variable
            reply = QMessageBox.question(self, "Add Variable", 
                "Would you like to add another independent variable?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                break

        if len(x_cols) == 0:
            QMessageBox.warning(self, "Warning", "No independent variables selected")
            return

        try:
            # Prepare data
            df = pd.DataFrame()
            df[y_col] = pd.to_numeric(self.data[y_col], errors='coerce')
            
            for col in x_cols:
                df[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Remove missing values
            df = df.dropna()

            if len(df) < len(x_cols) + 1:
                QMessageBox.warning(self, "Warning", 
                    "Insufficient data for regression analysis with these variables")
                return

            # Create formula for regression
            formula = f"{y_col} ~ " + " + ".join(x_cols)
            
            # Fit regression model
            model = sm.OLS.from_formula(formula, data=df).fit()
            
            # Create regression equation string
            equation = f"Y = {model.params.iloc[0]:.4f}"
            for i, col in enumerate(x_cols):
                coef = model.params.iloc[i + 1]
                equation += f" + {coef:.4f}*{col}"

            # Create correlation matrix
            correlation_matrix = df.corr()

            # Create report
            report = f"""Multiple Linear Regression Analysis

Dependent Variable (Y): {y_col}
Independent Variables (X): {', '.join(x_cols)}

Regression Equation:
{equation}

Model Summary:
R-squared: {model.rsquared:.4f}
Adjusted R-squared: {model.rsquared_adj:.4f}

Analysis of Variance:
{model.summary().tables[0]}

Coefficients:
{model.summary().tables[1]}

Correlation Matrix:
{correlation_matrix.to_string()}

Variance Inflation Factors (VIF):
"""
            # Calculate VIF for each independent variable
            X = sm.add_constant(df[x_cols])
            for i, col in enumerate(x_cols):
                y = X[col]
                X_temp = X.drop(col, axis=1)
                r2 = sm.OLS(y, X_temp).fit().rsquared
                vif = 1 / (1 - r2)
                report += f"{col}: {vif:.4f}\n"

            self.sessionWindow.setText(report)

            # Create residual plots
            n_plots = len(x_cols)
            fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
            if n_plots == 1:
                axes = [axes]

            residuals = model.resid
            fitted = model.fittedvalues

            for i, col in enumerate(x_cols):
                axes[i].scatter(df[col], residuals, alpha=0.5)
                axes[i].axhline(y=0, color='r', linestyle='--')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Residuals')
                axes[i].set_title(f'Residuals vs {col}')
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # Create normal probability plot of residuals
            from scipy import stats
            fig = plt.figure(figsize=(8, 6))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('Normal Probability Plot of Residuals')
            plt.grid(True)
            plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred during regression analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def createDOE(self):
        """Create Design of Experiments"""
        # Create dialog for DOE type selection
        doe_type, ok = QInputDialog.getItem(self, "Design of Experiments",
            "Select DOE Type:",
            ["2-level Factorial", "Fractional Factorial", "Response Surface"], 0, False)
        if not ok:
            return

        if doe_type == "2-level Factorial":
            self.create_factorial_design()
        elif doe_type == "Fractional Factorial":
            self.create_fractional_factorial()
        else:
            self.create_response_surface()

    def create_factorial_design(self):
        """Create 2-level factorial design"""
        # Get number of factors
        n_factors, ok = QInputDialog.getInt(self, "Factorial Design", 
            "Enter number of factors (2-6):", 2, 2, 6)
        if not ok:
            return

        # Get factor names and levels
        factors = []
        levels = []
        for i in range(n_factors):
            # Get factor name
            name, ok = QInputDialog.getText(self, f"Factor {i+1}", 
                f"Enter name for factor {i+1}:")
            if not ok:
                return
            factors.append(name)
            
            # Get factor levels
            low, ok = QInputDialog.getText(self, f"Factor {i+1} Low", 
                f"Enter low level for {name}:")
            if not ok:
                return
            high, ok = QInputDialog.getText(self, f"Factor {i+1} High", 
                f"Enter high level for {name}:")
            if not ok:
                return
            levels.append([low, high])

        # Create full factorial design
        n_runs = 2 ** n_factors
        design_matrix = []
        for i in range(n_runs):
            run = []
            for j in range(n_factors):
                # Convert run number to binary and use as factor levels
                level_idx = (i >> j) & 1
                run.append(levels[j][level_idx])
            design_matrix.append(run)

        # Create DataFrame with the design
        df = pd.DataFrame(design_matrix, columns=factors)
        df.insert(0, 'StdOrder', range(1, len(df) + 1))
        df.insert(1, 'RunOrder', np.random.permutation(len(df)) + 1)
        df['Response'] = ''  # Empty column for responses

        # Update the table with the design
        self.data = df
        self.updateTable()

        # Show design summary
        summary = f"""2-level Factorial Design Summary

Number of factors: {n_factors}
Number of runs: {n_runs}
Base design: Full factorial

Factors and Levels:
"""
        for i, factor in enumerate(factors):
            summary += f"{factor}: {levels[i][0]} | {levels[i][1]}\n"

        self.sessionWindow.setText(summary)

    def create_fractional_factorial(self):
        """Create fractional factorial design"""
        # Get number of factors
        n_factors, ok = QInputDialog.getInt(self, "Fractional Factorial Design", 
            "Enter number of factors (3-7):", 3, 3, 7)
        if not ok:
            return

        # Get resolution level
        resolution_options = []
        max_resolution = n_factors
        for r in range(3, max_resolution + 1):
            if 2**(n_factors - 1) >= n_factors:  # Modified condition
                resolution_options.append(f"Resolution {r}")
        
        if not resolution_options:
            QMessageBox.warning(self, "Warning", 
                "No valid resolution available for this number of factors")
            return

        resolution, ok = QInputDialog.getItem(self, "Fractional Factorial Design",
            "Select design resolution:", resolution_options, 0, False)
        if not ok:
            return
        
        resolution_level = int(resolution.split()[-1])

        # Get factor names and levels
        factors = []
        levels = []
        for i in range(n_factors):
            # Get factor name
            name, ok = QInputDialog.getText(self, f"Factor {i+1}", 
                f"Enter name for factor {i+1}:")
            if not ok:
                return
            factors.append(name)
            
            # Get factor levels
            low, ok = QInputDialog.getText(self, f"Factor {i+1} Low", 
                f"Enter low level for {name}:")
            if not ok:
                return
            high, ok = QInputDialog.getText(self, f"Factor {i+1} High", 
                f"Enter high level for {name}:")
            if not ok:
                return
            levels.append([low, high])

        # Calculate number of runs needed
        n_runs = 2**(n_factors - 1)  # Modified for half-fraction

        # Create basic design matrix
        basic_design = []
        for i in range(n_runs):
            run = []
            for j in range(n_factors - 1):  # Generate for all but one factor
                level_idx = (i >> j) & 1
                run.append(level_idx)
            
            # Generate the last factor based on the interaction of other factors
            last_factor = 0
            for j in range(n_factors - 1):
                last_factor ^= run[j]
            run.append(last_factor)
            
            basic_design.append(run)

        # Convert design to actual factor levels
        design_matrix = []
        for run in basic_design:
            actual_run = []
            for j, level_idx in enumerate(run):
                actual_run.append(levels[j][level_idx])
            design_matrix.append(actual_run)

        # Create DataFrame with the design
        df = pd.DataFrame(design_matrix, columns=factors)
        df.insert(0, 'StdOrder', range(1, len(df) + 1))
        df.insert(1, 'RunOrder', np.random.permutation(len(df)) + 1)
        df['Response'] = ''  # Empty column for responses

        # Update the table with the design
        self.data = df
        self.updateTable()

        # Show design summary
        summary = f"""Fractional Factorial Design Summary

Number of factors: {n_factors}
Resolution: {resolution}
Number of runs: {n_runs}
Fraction: 1/2

Factors and Levels:
"""
        for i, factor in enumerate(factors):
            summary += f"{factor}: {levels[i][0]} | {levels[i][1]}\n"

        if n_factors > 2:
            summary += "\nDesign Generator:\n"
            summary += f"{factors[-1]} = {' × '.join(factors[:-1])}\n"

        self.sessionWindow.setText(summary)

    def create_response_surface(self):
        """Create response surface design"""
        # Get design type
        design_type, ok = QInputDialog.getItem(self, "Response Surface Design",
            "Select design type:",
            ["Central Composite Design (CCD)", "Box-Behnken Design (BBD)"], 0, False)
        if not ok:
            return

        # Get number of factors
        min_factors = 2 if design_type.startswith("Central") else 3
        max_factors = 6 if design_type.startswith("Central") else 7
        n_factors, ok = QInputDialog.getInt(self, "Response Surface Design", 
            f"Enter number of factors ({min_factors}-{max_factors}):", 
            min_factors, min_factors, max_factors)
        if not ok:
            return

        # Get factor names and levels
        factors = []
        center_points = []
        ranges = []
        
        for i in range(n_factors):
            # Get factor name
            name, ok = QInputDialog.getText(self, f"Factor {i+1}", 
                f"Enter name for factor {i+1}:")
            if not ok:
                return
            factors.append(name)
            
            # Get factor center point and range
            center, ok = QInputDialog.getDouble(self, f"Factor {i+1} Center", 
                f"Enter center point for {name}:")
            if not ok:
                return
            center_points.append(center)
            
            range_val, ok = QInputDialog.getDouble(self, f"Factor {i+1} Range", 
                f"Enter range (±) for {name}:")
            if not ok:
                return
            ranges.append(range_val)

        # Get number of center points
        n_center, ok = QInputDialog.getInt(self, "Center Points", 
            "Enter number of center points:", 3, 1, 10)
        if not ok:
            return

        # Create design matrix based on design type
        if design_type.startswith("Central"):
            # Central Composite Design
            # Get alpha type
            alpha_type, ok = QInputDialog.getItem(self, "CCD Alpha",
                "Select alpha type:",
                ["Rotatable", "Orthogonal", "Face-centered"], 0, False)
            if not ok:
                return

            # Calculate alpha value
            if alpha_type == "Rotatable":
                alpha = (2**n_factors)**(1/4)
            elif alpha_type == "Face-centered":
                alpha = 1.0
            else:  # Orthogonal
                n_factorial = 2**n_factors
                n_axial = 2 * n_factors
                alpha = ((n_factorial * (1 + n_center/n_factorial))/n_axial)**(1/4)

            # Create factorial points
            factorial_points = []
            for i in range(2**n_factors):
                point = []
                for j in range(n_factors):
                    level = -1 if (i >> j) & 1 else 1
                    point.append(level)
                factorial_points.append(point)

            # Create axial points
            axial_points = []
            for i in range(n_factors):
                point_plus = [0] * n_factors
                point_minus = [0] * n_factors
                point_plus[i] = alpha
                point_minus[i] = -alpha
                axial_points.append(point_plus)
                axial_points.append(point_minus)

            # Create center points
            center_points_design = [[0] * n_factors for _ in range(n_center)]

            # Combine all points
            coded_design = factorial_points + axial_points + center_points_design

        else:
            # Box-Behnken Design
            if n_factors < 3:
                QMessageBox.warning(self, "Warning", 
                    "Box-Behnken design requires at least 3 factors")
                return

            # Create design points
            coded_design = []
            
            # Add factorial points for pairs of factors
            for i in range(n_factors - 1):
                for j in range(i + 1, n_factors):
                    for level_i in [-1, 1]:
                        for level_j in [-1, 1]:
                            point = [0] * n_factors
                            point[i] = level_i
                            point[j] = level_j
                            coded_design.append(point)

            # Add center points
            center_points_design = [[0] * n_factors for _ in range(n_center)]
            coded_design.extend(center_points_design)

        # Convert coded levels to actual values
        design_matrix = []
        for run in coded_design:
            actual_run = []
            for i, coded_level in enumerate(run):
                actual_value = center_points[i] + coded_level * ranges[i]
                actual_run.append(actual_value)
            design_matrix.append(actual_run)

        # Create DataFrame with the design
        df = pd.DataFrame(design_matrix, columns=factors)
        df.insert(0, 'StdOrder', range(1, len(df) + 1))
        df.insert(1, 'RunOrder', np.random.permutation(len(df)) + 1)
        df['Response'] = ''  # Empty column for responses

        # Add point type information
        if design_type.startswith("Central"):
            point_types = (['Factorial'] * 2**n_factors + 
                         ['Axial'] * (2 * n_factors) +
                         ['Center'] * n_center)
        else:  # Box-Behnken
            n_factorial = n_factors * (n_factors - 1) * 2
            point_types = ['Factorial'] * n_factorial + ['Center'] * n_center

        df['PointType'] = point_types

        # Update the table with the design
        self.data = df
        self.updateTable()

        # Show design summary
        summary = f"""Response Surface Design Summary

Design Type: {design_type}
Number of factors: {n_factors}
Number of runs: {len(df)}

Design Points:
Factorial points: {len([pt for pt in point_types if pt == 'Factorial'])}
{'Axial points: ' + str(len([pt for pt in point_types if pt == 'Axial'])) if design_type.startswith('Central') else ''}
Center points: {n_center}

Factors:
"""
        for i, factor in enumerate(factors):
            summary += f"{factor}:\n"
            summary += f"  Center: {center_points[i]}\n"
            summary += f"  Range: ±{ranges[i]}\n"

        if design_type.startswith("Central"):
            summary += f"\nAlpha Type: {alpha_type}"
            summary += f"\nAlpha Value: {alpha:.4f}"

        self.sessionWindow.setText(summary)

    def analyzeDOE(self):
        """Analyze Design of Experiments"""
        # Load the current data from the table
        self.loadDataFromTable()
        
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "No data available for analysis")
            return

        try:
            # Check if we have response data
            if 'Response' not in self.data.columns:
                QMessageBox.warning(self, "Warning", "No Response column found")
                return

            # Identify factor columns (exclude StdOrder, RunOrder, and Response)
            factor_cols = [col for col in self.data.columns 
                          if col not in ['StdOrder', 'RunOrder', 'Response']]

            if not factor_cols:
                QMessageBox.warning(self, "Warning", "No factor columns identified")
                return

            # Create a copy of the data to avoid modifying the original
            analysis_data = self.data.copy()

            # Convert response to numeric, dropping any non-numeric values
            analysis_data['Response'] = pd.to_numeric(analysis_data['Response'].astype(str).str.strip(), errors='coerce')
            
            # Drop any rows with missing response values
            analysis_data = analysis_data.dropna(subset=['Response'])

            if len(analysis_data) == 0:
                QMessageBox.warning(self, "Warning", "No valid response data after conversion")
                return

            # Create design matrix with coded values (-1, 1)
            design_matrix = pd.DataFrame()
            for col in factor_cols:
                col_data = pd.to_numeric(analysis_data[col], errors='coerce')
                col_min = col_data.min()
                col_max = col_data.max()
                if col_max > col_min:  # Ensure we don't divide by zero
                    design_matrix[col] = -1 + 2 * (col_data - col_min) / (col_max - col_min)
                else:
                    QMessageBox.warning(self, "Warning", f"Factor {col} has no variation")
                    return

            # Add constant term for the regression
            X = sm.add_constant(design_matrix)
            y = analysis_data['Response']

            # Fit the model
            model = sm.OLS(y, X).fit()

            # Calculate effects (multiply coefficients by 2 for proper effect size)
            effects = pd.Series(model.params[1:] * 2, index=factor_cols)

            # Create analysis report
            report = f"""Design of Experiments Analysis

Model Summary:
R-squared: {model.rsquared:.4f}
Adjusted R-squared: {model.rsquared_adj:.4f}
F-statistic: {model.fvalue:.4f}
Prob(F-statistic): {model.f_pvalue:.4f}

Effects:
"""
            for factor, effect in effects.items():
                report += f"{factor}: {effect:.4f}\n"

            report += f"""
Analysis of Variance:
{model.summary().tables[2]}

Parameter Estimates:
{model.summary().tables[1]}
"""
            self.sessionWindow.setText(report)

            # Create effects plot
            plt.figure(figsize=(10, 6))
            effects.plot(kind='bar')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Main Effects Plot')
            plt.xlabel('Factors')
            plt.ylabel('Effect Size')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # Create normal probability plot of effects
            fig = plt.figure(figsize=(8, 6))
            stats.probplot(effects, dist="norm", plot=plt)
            plt.title('Normal Probability Plot of Effects')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred during DOE analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def individualChart(self):
        """Create individual control chart"""
        # Select data column
        col = self.selectColumnDialog()
        if not col:
            return
        
        # Get the data
        data = pd.to_numeric(self.data[col], errors='coerce').dropna()
        
        if len(data) < 2:
            QMessageBox.warning(self, "Warning", "Need at least 2 points for I-MR charts")
            return
        
        # Calculate moving ranges
        moving_range = np.abs(np.diff(data))
        
        # Calculate control limits for Individual chart
        x_mean = np.mean(data)
        mr_mean = np.mean(moving_range)
        
        # Constants for control limits
        E2 = 2.66  # Constant for control limits with n=2
        
        # Calculate limits for Individual chart
        i_ucl = x_mean + (E2 * mr_mean / 1.128)
        i_lcl = x_mean - (E2 * mr_mean / 1.128)
        
        # Create the Individual chart
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(data, marker='o', linestyle='-', color='blue')
        plt.axhline(y=x_mean, color='green', linestyle='-', label='Mean')
        plt.axhline(y=i_ucl, color='red', linestyle='--', label='UCL')
        plt.axhline(y=i_lcl, color='red', linestyle='--', label='LCL')
        plt.title(f'Individual Chart for {col}')
        plt.ylabel('Individual Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create report
        report = f"""Individual Chart Analysis for {col}

Individual Chart Statistics:
Mean: {x_mean:.3f}
UCL: {i_ucl:.3f}
LCL: {i_lcl:.3f}

Number of Points: {len(data)}
Points Outside Control Limits: {sum((data > i_ucl) | (data < i_lcl))}

Note: Control limits are based on ±3 sigma (calculated using moving ranges)
"""
        self.sessionWindow.setText(report)
        plt.show()

    def movingRangeChart(self):
        """Create moving range chart"""
        # Select data column
        col = self.selectColumnDialog()
        if not col:
            return
        
        # Get the data
        data = pd.to_numeric(self.data[col], errors='coerce').dropna()
        
        if len(data) < 2:
            QMessageBox.warning(self, "Warning", "Need at least 2 points for Moving Range chart")
            return
        
        # Calculate moving ranges
        moving_range = np.abs(np.diff(data))
        
        # Calculate control limits for Moving Range chart
        mr_mean = np.mean(moving_range)
        
        # Constants for Moving Range chart
        D3 = 0  # Lower control limit constant for n=2
        D4 = 3.267  # Upper control limit constant for n=2
        
        # Calculate limits for Moving Range chart
        mr_ucl = D4 * mr_mean
        mr_lcl = D3 * mr_mean
        
        # Create the Moving Range chart
        plt.figure(figsize=(12, 6))
        plt.plot(moving_range, marker='o', linestyle='-', color='blue')
        plt.axhline(y=mr_mean, color='green', linestyle='-', label='Mean')
        plt.axhline(y=mr_ucl, color='r', linestyle='--', label='UCL')
        plt.axhline(y=mr_lcl, color='r', linestyle='--', label='LCL')
        plt.title(f'Moving Range Chart for {col}')
        plt.ylabel('Moving Range')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create report
        report = f"""Moving Range Chart Analysis for {col}

Moving Range Statistics:
Mean MR: {mr_mean:.3f}
UCL: {mr_ucl:.3f}
LCL: {mr_lcl:.3f}

Number of Ranges: {len(moving_range)}
Ranges Outside Control Limits: {sum((moving_range > mr_ucl) | (moving_range < mr_lcl))}

Note: Control limits are based on ±3 sigma
"""
        self.sessionWindow.setText(report)
        plt.show()

    def msaAnalysis(self):
        """Perform Measurement System Analysis"""
        # Create dialog for MSA type selection
        msa_type, ok = QInputDialog.getItem(self, "Measurement System Analysis",
            "Select MSA Type:",
            ["Gage R&R Study", "Linearity Study", "Bias Study", "Stability Study"], 0, False)
        if not ok:
            return

        if msa_type == "Gage R&R Study":
            self.gageRR()
        elif msa_type == "Linearity Study":
            self.linearityStudy()
        elif msa_type == "Bias Study":
            self.biasStudy()
        else:  # Stability Study
            self.stabilityStudy()

    def gageRR(self):
        """Perform Gage R&R Study"""
        # Get measurement column
        measurement_col = self.selectColumnDialog()
        if not measurement_col:
            return

        # Get operator column
        operator_col = self.selectColumnDialog()
        if not operator_col:
            return

        # Get part column
        part_col = self.selectColumnDialog()
        if not part_col:
            return

        try:
            # Create DataFrame for analysis
            df = pd.DataFrame({
                'Measurement': pd.to_numeric(self.data[measurement_col], errors='coerce'),
                'Operator': self.data[operator_col],
                'Part': self.data[part_col]
            })

            # Remove missing values
            df = df.dropna()

            if len(df) < 2:
                QMessageBox.warning(self, "Warning", "Insufficient data for Gage R&R analysis")
                return

            # Calculate overall statistics
            total_mean = df['Measurement'].mean()
            total_std = df['Measurement'].std()

            # Calculate components of variation
            operators = df['Operator'].unique()
            parts = df['Part'].unique()
            n_operators = len(operators)
            n_parts = len(parts)
            n_measurements = len(df) / (n_operators * n_parts)

            # Calculate operator variation
            operator_means = df.groupby('Operator')['Measurement'].mean()
            operator_variation = operator_means.std() ** 2

            # Calculate part variation
            part_means = df.groupby('Part')['Measurement'].mean()
            part_variation = part_means.std() ** 2

            # Calculate repeatability (equipment variation)
            residuals = []
            for operator in operators:
                for part in parts:
                    measurements = df[(df['Operator'] == operator) & (df['Part'] == part)]['Measurement']
                    mean = measurements.mean()
                    residuals.extend(measurements - mean)
            repeatability = np.std(residuals) ** 2

            # Calculate total variation
            total_variation = part_variation + operator_variation + repeatability

            # Calculate %Contribution and %Study Variation
            contribution = {
                'Repeatability': (repeatability / total_variation) * 100,
                'Reproducibility': (operator_variation / total_variation) * 100,
                'Part-to-Part': (part_variation / total_variation) * 100
            }

            study_var = {
                'Repeatability': np.sqrt(repeatability) * 6,
                'Reproducibility': np.sqrt(operator_variation) * 6,
                'Part-to-Part': np.sqrt(part_variation) * 6,
                'Total': np.sqrt(total_variation) * 6
            }

            # Create report
            report = f"""Gage R&R Study Results

Study Information:
Number of Operators: {n_operators}
Number of Parts: {n_parts}
Number of Replicates: {int(n_measurements)}

Overall Statistics:
Mean: {total_mean:.3f}
Standard Deviation: {total_std:.3f}

Variance Components:
Source          %Contribution  Study Var  %Study Var
Total Gage R&R  {(contribution['Repeatability'] + contribution['Reproducibility']):.1f}%  {(study_var['Repeatability'] + study_var['Reproducibility']):.3f}  {((study_var['Repeatability'] + study_var['Reproducibility'])/study_var['Total']*100):.1f}%
  Repeatability {contribution['Repeatability']:.1f}%  {study_var['Repeatability']:.3f}  {(study_var['Repeatability']/study_var['Total']*100):.1f}%
  Reproducibility {contribution['Reproducibility']:.1f}%  {study_var['Reproducibility']:.3f}  {(study_var['Reproducibility']/study_var['Total']*100):.1f}%
Part-to-Part    {contribution['Part-to-Part']:.1f}%  {study_var['Part-to-Part']:.3f}  {(study_var['Part-to-Part']/study_var['Total']*100):.1f}%
Total Variation 100%  {study_var['Total']:.3f}  100%

Number of Distinct Categories: {int(np.sqrt(2 * (1 - (contribution['Repeatability'] + contribution['Reproducibility'])/100)))}

Assessment:
"""
            # Add assessment based on results
            total_gage_rr = contribution['Repeatability'] + contribution['Reproducibility']
            if total_gage_rr < 10:
                report += "Measurement system is acceptable."
            elif total_gage_rr < 30:
                report += "Measurement system may be acceptable depending on application."
            else:
                report += "Measurement system needs improvement."

            self.sessionWindow.setText(report)

            # Create visualizations
            # 1. Components of Variation
            plt.figure(figsize=(10, 6))
            plt.bar(['Repeatability', 'Reproducibility', 'Part-to-Part'],
                    [contribution['Repeatability'], contribution['Reproducibility'], contribution['Part-to-Part']])
            plt.title('Components of Variation')
            plt.ylabel('Percent Contribution (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # 2. Measurement by Operator
            plt.figure(figsize=(10, 6))
            for operator in operators:
                operator_data = df[df['Operator'] == operator]
                plt.plot(operator_data['Part'], operator_data['Measurement'], 
                        marker='o', linestyle='-', label=f'Operator {operator}')
            plt.title('Measurement by Operator')
            plt.xlabel('Part')
            plt.ylabel('Measurement')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred during Gage R&R analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def linearityStudy(self):
        """Perform measurement system linearity study"""
        # Get reference value column
        reference_col = self.selectColumnDialog()
        if not reference_col:
            return

        # Get measurement column
        measurement_col = self.selectColumnDialog()
        if not measurement_col:
            return

        try:
            # Create DataFrame for analysis
            df = pd.DataFrame({
                'Reference': pd.to_numeric(self.data[reference_col], errors='coerce'),
                'Measurement': pd.to_numeric(self.data[measurement_col], errors='coerce')
            })

            # Remove missing values
            df = df.dropna()

            if len(df) < 2:
                QMessageBox.warning(self, "Warning", "Insufficient data for linearity analysis")
                return

            # Calculate bias at each reference point
            df['Bias'] = df['Measurement'] - df['Reference']

            # Perform linear regression on bias vs reference
            model = sm.OLS.from_formula('Bias ~ Reference', data=df).fit()

            # Calculate predicted bias values
            df['Predicted_Bias'] = model.predict(df)

            # Calculate confidence intervals
            prediction = model.get_prediction(df)
            df['CI_Lower'] = prediction.conf_int()[:, 0]
            df['CI_Upper'] = prediction.conf_int()[:, 1]

            # Create report
            report = f"""Linearity Study Results

Study Information:
Number of Reference Points: {len(df['Reference'].unique())}
Total Measurements: {len(df)}

Regression Analysis:
Intercept: {model.params[0]:.4f}
Slope: {model.params[1]:.4f}
R-squared: {model.rsquared:.4f}

Hypothesis Tests:
H0: Slope = 0
p-value: {model.pvalues[1]:.4f}

H0: Intercept = 0
p-value: {model.pvalues[0]:.4f}

Average Bias: {df['Bias'].mean():.4f}
Standard Deviation of Bias: {df['Bias'].std():.4f}

Assessment:
"""
            # Add assessment based on results
            if abs(model.params[1]) < 0.01 and model.pvalues[1] > 0.05:
                report += "No significant linearity issue detected."
            else:
                report += "Significant linearity effect detected. System may need calibration."

            if abs(df['Bias'].mean()) < df['Bias'].std():
                report += "\nOverall bias is within expected variation."
            else:
                report += "\nSignificant bias detected. System may need adjustment."

            self.sessionWindow.setText(report)

            # Create visualizations
            # 1. Bias Plot
            plt.figure(figsize=(10, 6))
            plt.scatter(df['Reference'], df['Bias'], alpha=0.5, label='Observed Bias')
            plt.plot(df['Reference'], df['Predicted_Bias'], 'r-', label='Fitted Line')
            plt.fill_between(df['Reference'], df['CI_Lower'], df['CI_Upper'], 
                           color='gray', alpha=0.2, label='95% Confidence Interval')
            plt.axhline(y=0, color='g', linestyle='--', label='Zero Bias')
            plt.title('Linearity Study: Bias Plot')
            plt.xlabel('Reference Value')
            plt.ylabel('Bias (Measurement - Reference)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            # 2. Measurement vs Reference Plot
            plt.figure(figsize=(10, 6))
            plt.scatter(df['Reference'], df['Measurement'], alpha=0.5)
            min_val = min(df['Reference'].min(), df['Measurement'].min())
            max_val = max(df['Reference'].max(), df['Measurement'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement')
            plt.title('Measurement vs Reference')
            plt.xlabel('Reference Value')
            plt.ylabel('Measurement')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred during linearity analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def fishboneDiagram(self):
        """Create and edit fishbone diagram"""
        try:
            # Create dialog for fishbone diagram
            dialog = QDialog(self)
            dialog.setWindowTitle("Fishbone Diagram Creator")
            dialog.resize(800, 600)

            # Create layout
            layout = QVBoxLayout()

            # Add problem statement input
            problem_label = QLabel("Problem Statement:")
            problem_input = QLineEdit()
            layout.addWidget(problem_label)
            layout.addWidget(problem_input)

            # Create category inputs
            categories = ['Materials', 'Methods', 'Machines', 'Manpower', 'Measurement', 'Environment']
            category_inputs = {}

            for category in categories:
                # Create group box for category
                group = QGroupBox(category)
                group_layout = QVBoxLayout()

                # Add 5 cause input fields for each category
                cause_inputs = []
                for i in range(5):
                    cause_input = QLineEdit()
                    cause_input.setPlaceholderText(f"Enter cause {i+1}")
                    group_layout.addWidget(cause_input)
                    cause_inputs.append(cause_input)

                group.setLayout(group_layout)
                layout.addWidget(group)
                category_inputs[category] = cause_inputs

            # Add create button
            create_button = QPushButton("Create Diagram")
            layout.addWidget(create_button)

            dialog.setLayout(layout)

            def create_diagram():
                # Get problem statement
                problem = problem_input.text()
                if not problem:
                    QMessageBox.warning(dialog, "Warning", "Please enter a problem statement")
                    return

                # Collect causes for each category
                causes = {}
                for category, inputs in category_inputs.items():
                    causes[category] = [input.text() for input in inputs if input.text()]

                # Create the fishbone diagram
                plt.figure(figsize=(15, 10))
                
                # Draw main arrow and problem box
                plt.plot([0, 10], [5, 5], 'k-', linewidth=2)  # Main arrow
                plt.text(10.2, 5, problem, ha='left', va='center', fontsize=12, fontweight='bold')

                # Position categories and their causes
                y_positions = [2, 3, 4, 6, 7, 8]
                for i, (category, category_causes) in enumerate(causes.items()):
                    y = y_positions[i]
                    # Draw category arrow
                    if y < 5:  # Below center line
                        plt.plot([2+i, 5], [y, 5], 'k-')
                    else:  # Above center line
                        plt.plot([2+(5-i), 5], [y, 5], 'k-')
                    
                    # Add category label
                    plt.text(2+i if y < 5 else 2+(5-i), y, category, 
                            ha='right' if y < 5 else 'left', 
                            va='bottom' if y < 5 else 'top',
                            fontsize=10)

                    # Add causes
                    for j, cause in enumerate(category_causes):
                        if cause:
                            x_offset = 0.5 * j
                            if y < 5:  # Below center line
                                plt.plot([2+i+x_offset, 2+i+x_offset+0.5], 
                                       [y+0.2*j, y+0.2*j+0.2], 'k-')
                                plt.text(2+i+x_offset, y+0.2*j, cause, 
                                       ha='right', va='bottom', fontsize=8)
                            else:  # Above center line
                                plt.plot([2+(5-i)+x_offset, 2+(5-i)+x_offset+0.5],
                                       [y-0.2*j, y-0.2*j-0.2], 'k-')
                                plt.text(2+(5-i)+x_offset, y-0.2*j, cause,
                                       ha='left', va='top', fontsize=8)

                plt.axis('off')
                plt.title('Fishbone (Ishikawa) Diagram', pad=20)
                plt.tight_layout()
                plt.show()

                # Create report
                report = f"""Fishbone Diagram Analysis

Problem Statement: {problem}

Categories and Causes:
"""
                for category, category_causes in causes.items():
                    report += f"\n{category}:"
                    for cause in category_causes:
                        if cause:
                            report += f"\n  - {cause}"

                self.sessionWindow.setText(report)

            create_button.clicked.connect(create_diagram)
            dialog.exec()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred while creating the fishbone diagram:\n{str(e)}")

    def fmeaTemplate(self):
        """Create and manage FMEA template"""
        try:
            # Create dialog for FMEA
            dialog = QDialog(self)
            dialog.setWindowTitle("FMEA Template")
            dialog.resize(1000, 800)

            # Create layout
            layout = QVBoxLayout()

            # Add header information
            header_group = QGroupBox("FMEA Header")
            header_layout = QFormLayout()
            
            process_input = QLineEdit()
            responsible_input = QLineEdit()
            date_input = QLineEdit()
            
            header_layout.addRow("Process/Product:", process_input)
            header_layout.addRow("Team/Responsible:", responsible_input)
            header_layout.addRow("Date:", date_input)
            
            header_group.setLayout(header_layout)
            layout.addWidget(header_group)

            # Create table for FMEA entries
            table = QTableWidget()
            table.setColumnCount(10)
            table.setRowCount(10)  # Initial rows, can be expanded
            
            # Set column headers
            headers = [
                "Process Step",
                "Potential Failure Mode",
                "Potential Effects",
                "Severity (1-10)",
                "Potential Causes",
                "Occurrence (1-10)",
                "Current Controls",
                "Detection (1-10)",
                "RPN",
                "Recommended Actions"
            ]
            table.setHorizontalHeaderLabels(headers)
            
            # Set column widths
            table.setColumnWidth(0, 120)  # Process Step
            table.setColumnWidth(1, 120)  # Failure Mode
            table.setColumnWidth(2, 120)  # Effects
            table.setColumnWidth(3, 80)   # Severity
            table.setColumnWidth(4, 120)  # Causes
            table.setColumnWidth(5, 80)   # Occurrence
            table.setColumnWidth(6, 120)  # Controls
            table.setColumnWidth(7, 80)   # Detection
            table.setColumnWidth(8, 80)   # RPN
            table.setColumnWidth(9, 120)  # Actions

            layout.addWidget(table)

            # Add buttons
            button_layout = QHBoxLayout()
            
            add_row_button = QPushButton("Add Row")
            calculate_button = QPushButton("Calculate RPN")
            save_button = QPushButton("Save FMEA")
            export_button = QPushButton("Export Report")
            save_report_button = QPushButton("Save Report")  # New button
            
            button_layout.addWidget(add_row_button)
            button_layout.addWidget(calculate_button)
            button_layout.addWidget(save_button)
            button_layout.addWidget(export_button)
            button_layout.addWidget(save_report_button)  # Add new button
            
            layout.addLayout(button_layout)

            dialog.setLayout(layout)

            # Add functionality to buttons
            def add_row():
                current_row = table.rowCount()
                table.insertRow(current_row)

            def calculate_rpn():
                for row in range(table.rowCount()):
                    try:
                        # Get severity, occurrence, and detection values
                        severity = float(table.item(row, 3).text() if table.item(row, 3) else 0)
                        occurrence = float(table.item(row, 5).text() if table.item(row, 5) else 0)
                        detection = float(table.item(row, 7).text() if table.item(row, 7) else 0)
                        
                        # Calculate RPN
                        rpn = severity * occurrence * detection
                        
                        # Update RPN cell
                        rpn_item = QTableWidgetItem(str(int(rpn)))
                        table.setItem(row, 8, rpn_item)
                        
                        # Highlight high RPNs (>100) in red
                        if rpn > 100:
                            rpn_item.setBackground(Qt.GlobalColor.red)
                    except (ValueError, AttributeError):
                        continue

            def save_fmea():
                # Create DataFrame from table data
                data = []
                for row in range(table.rowCount()):
                    row_data = []
                    for col in range(table.columnCount()):
                        item = table.item(row, col)
                        row_data.append(item.text() if item else "")
                    data.append(row_data)
                
                df = pd.DataFrame(data, columns=headers)
                
                # Save to CSV
                filename, _ = QFileDialog.getSaveFileName(dialog, "Save FMEA", "", "CSV Files (*.csv)")
                if filename:
                    df.to_csv(filename, index=False)
                    QMessageBox.information(dialog, "Success", "FMEA saved successfully!")

            def generate_report():
                # Generate detailed report
                report = f"""FMEA Analysis Report

Process/Product: {process_input.text()}
Team/Responsible: {responsible_input.text()}
Date: {date_input.text()}

Summary of Failure Modes and Actions:
"""
                # Add table contents to report
                for row in range(table.rowCount()):
                    process_step = table.item(row, 0)
                    failure_mode = table.item(row, 1)
                    effects = table.item(row, 2)
                    severity = table.item(row, 3)
                    causes = table.item(row, 4)
                    occurrence = table.item(row, 5)
                    controls = table.item(row, 6)
                    detection = table.item(row, 7)
                    rpn = table.item(row, 8)
                    actions = table.item(row, 9)
                    
                    if process_step and process_step.text():
                        report += f"\nProcess Step: {process_step.text()}"
                        report += f"\nFailure Mode: {failure_mode.text() if failure_mode else 'Not specified'}"
                        report += f"\nPotential Effects: {effects.text() if effects else 'Not specified'}"
                        report += f"\nSeverity: {severity.text() if severity else 'Not specified'}"
                        report += f"\nPotential Causes: {causes.text() if causes else 'Not specified'}"
                        report += f"\nOccurrence: {occurrence.text() if occurrence else 'Not specified'}"
                        report += f"\nCurrent Controls: {controls.text() if controls else 'Not specified'}"
                        report += f"\nDetection: {detection.text() if detection else 'Not specified'}"
                        report += f"\nRPN: {rpn.text() if rpn else 'Not calculated'}"
                        report += f"\nRecommended Actions: {actions.text() if actions else 'None'}"
                        report += "\n" + "-"*50
                return report

            def export_report():
                report = generate_report()
                # Display report in session window
                self.sessionWindow.clear()  # Clear previous content
                self.sessionWindow.setText(report)
                QMessageBox.information(dialog, "Report Generated", "Report has been generated in the session window.")

            def save_report():
                report = generate_report()
                # Save report to text file
                filename, _ = QFileDialog.getSaveFileName(dialog, "Save Report", "", "Text Files (*.txt)")
                if filename:
                    with open(filename, 'w') as f:
                        f.write(report)
                    QMessageBox.information(dialog, "Success", "Report saved successfully!")

            # Connect buttons to functions
            add_row_button.clicked.connect(add_row)
            calculate_button.clicked.connect(calculate_rpn)
            save_button.clicked.connect(save_fmea)
            export_button.clicked.connect(export_report)
            save_report_button.clicked.connect(save_report)  # Connect new button

            # Add help text
            help_text = """
Rating Guidelines:
Severity (1-10): Impact of failure on customer/process
- 1: No effect
- 5: Moderate effect
- 10: Hazardous effect

Occurrence (1-10): Likelihood of failure occurring
- 1: Very unlikely
- 5: Occasional failure
- 10: Failure is almost inevitable

Detection (1-10): Ability to detect failure before it reaches customer
- 1: Certain detection
- 5: Moderate detection
- 10: No detection

RPN (Risk Priority Number) = Severity × Occurrence × Detection
RPN > 100 indicates high priority items needing immediate attention.
"""
            help_label = QLabel(help_text)
            layout.addWidget(help_label)

            dialog.exec()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred while creating the FMEA template:\n{str(e)}")

    def sigmaLevelCalc(self):
        """Calculate Sigma Level from DPMO"""
        dpmo, ok = QInputDialog.getDouble(self, "Sigma Calculator", "Enter DPMO:", 0, 0)
        if not ok:
            return
            
        sigma = dpmo_to_sigma(dpmo)
        self.sessionWindow.setText(f"Sigma Level: {sigma:.2f}")

    def yieldAnalysis(self):
        """Calculate process yield based on defects and opportunities"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Process Yield Analysis")
        layout = QFormLayout()

        # Input fields
        units_input = QLineEdit()
        defects_input = QLineEdit()
        opportunities_input = QLineEdit()

        layout.addRow("Number of Units:", units_input)
        layout.addRow("Total Defects:", defects_input)
        layout.addRow("Opportunities per Unit:", opportunities_input)

        # Results labels
        yield_label = QLabel()
        dpmo_label = QLabel()
        sigma_label = QLabel()

        # Calculate button
        calc_button = QPushButton("Calculate")

        def calculate_yield():
            try:
                units = float(units_input.text())
                defects = float(defects_input.text())
                opportunities = float(opportunities_input.text())

                # Calculate DPMO
                dpmo = calculate_dpmo(defects, opportunities, units)
                
                # Calculate process yield
                total_opportunities = units * opportunities
                process_yield = ((total_opportunities - defects) / total_opportunities) * 100
                
                # Calculate sigma level
                sigma_level = dpmo_to_sigma(dpmo)

                # Update labels
                yield_label.setText(f"Process Yield: {process_yield:.2f}%")
                dpmo_label.setText(f"DPMO: {dpmo:.2f}")
                sigma_label.setText(f"Sigma Level: {sigma_level:.2f}")

                # Add results to session window
                report = f"""Process Yield Analysis Results:
------------------------
Number of Units: {units}
Total Defects: {defects}
Opportunities per Unit: {opportunities}
------------------------
Process Yield: {process_yield:.2f}%
DPMO: {dpmo:.2f}
Sigma Level: {sigma_level:.2f}
"""
                self.sessionWindow.append(report)

            except ValueError:
                QMessageBox.warning(dialog, "Error", "Please enter valid numbers")

        calc_button.clicked.connect(calculate_yield)

        # Add results to layout
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        results_layout.addWidget(yield_label)
        results_layout.addWidget(dpmo_label)
        results_layout.addWidget(sigma_label)
        results_group.setLayout(results_layout)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(calc_button)
        main_layout.addWidget(results_group)

        dialog.setLayout(main_layout)
        dialog.exec()

    def biasStudy(self):
        """Perform measurement system bias study"""
        # Get reference value
        reference_value, ok = QInputDialog.getDouble(self, "Bias Study", 
            "Enter reference (standard) value:", 0.0)
        if not ok:
            return

        # Get measurement column
        measurement_col = self.selectColumnDialog()
        if not measurement_col:
            return

        try:
            # Get measurements
            measurements = pd.to_numeric(self.data[measurement_col], errors='coerce').dropna()

            if len(measurements) < 2:
                QMessageBox.warning(self, "Warning", "Insufficient data for bias analysis")
                return

            # Calculate basic statistics
            mean = np.mean(measurements)
            std_dev = np.std(measurements, ddof=1)
            bias = mean - reference_value
            
            # Perform t-test to check if bias is significant
            t_stat, p_value = stats.ttest_1samp(measurements, reference_value)
            
            # Calculate confidence interval for bias
            ci = stats.t.interval(0.95, len(measurements)-1, loc=mean, scale=std_dev/np.sqrt(len(measurements)))
            
            # Calculate capability indices
            Cg = (0.2 * (reference_value))/(6 * std_dev)  # Precision to tolerance ratio
            Cgk = (0.1 * (reference_value) - abs(bias))/(3 * std_dev)  # Accuracy to tolerance ratio

            # Create report
            report = f"""Bias Study Results

Reference Value: {reference_value:.4f}

Basic Statistics:
Number of Measurements: {len(measurements)}
Mean: {mean:.4f}
Standard Deviation: {std_dev:.4f}
Bias: {bias:.4f}

Hypothesis Test (H0: Bias = 0):
t-statistic: {t_stat:.4f}
p-value: {p_value:.4f}

95% Confidence Interval for Mean:
Lower: {ci[0]:.4f}
Upper: {ci[1]:.4f}

Capability Indices:
Cg (Precision/Tolerance): {Cg:.4f}
Cgk (Accuracy/Tolerance): {Cgk:.4f}

Assessment:
"""
            # Add assessment based on results
            if abs(bias) < std_dev:
                report += "Bias is within one standard deviation - acceptable.\n"
            else:
                report += "Bias is larger than one standard deviation - may need adjustment.\n"

            if p_value < 0.05:
                report += "Bias is statistically significant (p < 0.05).\n"
            else:
                report += "No significant bias detected (p >= 0.05).\n"

            if Cg >= 1.33 and Cgk >= 1.33:
                report += "Measurement system is capable (Cg & Cgk >= 1.33)."
            else:
                report += "Measurement system may need improvement (Cg or Cgk < 1.33)."

            self.sessionWindow.setText(report)

            # Create visualizations
            # 1. Individual measurements with reference
            plt.figure(figsize=(10, 6))
            plt.plot(measurements, marker='o', linestyle='-', label='Measurements')
            plt.axhline(y=reference_value, color='r', linestyle='--', label='Reference')
            plt.axhline(y=mean, color='g', linestyle='-', label='Mean')
            plt.fill_between(range(len(measurements)), 
                           [ci[0]]*len(measurements), 
                           [ci[1]]*len(measurements), 
                           color='gray', alpha=0.2, label='95% CI')
            plt.title('Bias Study: Individual Measurements')
            plt.xlabel('Measurement Number')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            # 2. Histogram with reference
            plt.figure(figsize=(10, 6))
            plt.hist(measurements, bins='auto', density=True, alpha=0.7, label='Measurements')
            plt.axvline(x=reference_value, color='r', linestyle='--', label='Reference')
            plt.axvline(x=mean, color='g', linestyle='-', label='Mean')
            
            # Add normal curve
            x = np.linspace(min(measurements), max(measurements), 100)
            y = stats.norm.pdf(x, mean, std_dev)
            plt.plot(x, y, 'b-', label='Normal Dist.')
            
            plt.title('Bias Study: Distribution of Measurements')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred during bias analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def stabilityStudy(self):
        """Perform measurement system stability study"""
        # Get measurement column
        measurement_col = self.selectColumnDialog()
        if not measurement_col:
            return

        # Get time/date column
        time_col = self.selectColumnDialog()
        if not time_col:
            return

        try:
            # Create DataFrame for analysis
            df = pd.DataFrame({
                'Measurement': pd.to_numeric(self.data[measurement_col], errors='coerce'),
                'Time': pd.to_datetime(self.data[time_col], errors='coerce')
            })

            # Remove missing values
            df = df.dropna()
            df = df.sort_values('Time')

            if len(df) < 2:
                QMessageBox.warning(self, "Warning", "Insufficient data for stability analysis")
                return

            # Calculate control limits
            mean = df['Measurement'].mean()
            std_dev = df['Measurement'].std(ddof=1)
            ucl = mean + 3 * std_dev
            lcl = mean - 3 * std_dev

            # Calculate moving range
            moving_range = np.abs(df['Measurement'].diff())
            mr_mean = moving_range.mean()
            mr_ucl = 3.267 * mr_mean  # D4 = 3.267 for n=2
            mr_lcl = 0  # D3 = 0 for n=2

            # Check for out of control points
            out_of_control = df[
                (df['Measurement'] > ucl) | 
                (df['Measurement'] < lcl)
            ]
            mr_out_of_control = moving_range[
                (moving_range > mr_ucl) | 
                (moving_range < mr_lcl)
            ]

            # Perform trend analysis
            # Calculate correlation between measurements and time
            time_nums = (df['Time'] - df['Time'].min()).dt.total_seconds()
            correlation = np.corrcoef(time_nums, df['Measurement'])[0,1]

            # Create report
            report = f"""Stability Study Results

Basic Statistics:
Number of Measurements: {len(df)}
Time Period: {df['Time'].min().strftime('%Y-%m-%d')} to {df['Time'].max().strftime('%Y-%m-%d')}
Mean: {mean:.4f}
Standard Deviation: {std_dev:.4f}

Control Limits (Individual Measurements):
UCL: {ucl:.4f}
LCL: {lcl:.4f}
Points Outside Limits: {len(out_of_control)}

Moving Range Analysis:
Average MR: {mr_mean:.4f}
MR UCL: {mr_ucl:.4f}
MR Points Outside Limits: {len(mr_out_of_control)}

Trend Analysis:
Time-Measurement Correlation: {correlation:.4f}

Assessment:
"""
            # Add assessment based on results
            if len(out_of_control) == 0 and len(mr_out_of_control) == 0:
                report += "Process appears to be in statistical control.\n"
            else:
                report += "Process shows signs of instability.\n"

            if abs(correlation) > 0.5:
                report += "Significant trend detected over time.\n"
            else:
                report += "No significant trend detected.\n"

            self.sessionWindow.setText(report)

            # Create visualizations
            # 1. Individual measurements control chart
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(df['Time'], df['Measurement'], marker='o', linestyle='-', label='Measurements')
            plt.axhline(y=mean, color='g', linestyle='-', label='Mean')
            plt.axhline(y=ucl, color='r', linestyle='--', label='UCL')
            plt.axhline(y=lcl, color='r', linestyle='--', label='LCL')
            plt.title('Stability Study: Individual Measurements')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 2. Moving Range chart
            plt.subplot(2, 1, 2)
            plt.plot(df['Time'][1:], moving_range[1:], marker='o', linestyle='-', label='Moving Range')
            plt.axhline(y=mr_mean, color='g', linestyle='-', label='MR Mean')
            plt.axhline(y=mr_ucl, color='r', linestyle='--', label='MR UCL')
            plt.axhline(y=mr_lcl, color='r', linestyle='--', label='MR LCL')
            plt.title('Moving Range Chart')
            plt.xlabel('Time')
            plt.ylabel('Moving Range')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred during stability analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def chiSquareTests(self):
        """Perform Chi-Square analysis"""
        # Create dialog for test selection
        test_type, ok = QInputDialog.getItem(self, "Chi-Square Analysis",
            "Select Test Type:",
            ["Goodness of Fit", "Test of Independence", "Test of Homogeneity"], 0, False)
        if not ok:
            return

        if test_type == "Goodness of Fit":
            self.chiSquareGoodnessOfFit()
        elif test_type == "Test of Independence":
            self.chiSquareIndependence()
        else:
            self.chiSquareHomogeneity()

    def chiSquareGoodnessOfFit(self):
        """Perform Chi-Square Goodness of Fit test"""
        # Get observed frequencies column
        observed_col = self.selectColumnDialog()
        if not observed_col:
            return

        # Get expected probabilities
        prob_type, ok = QInputDialog.getItem(self, "Expected Probabilities",
            "Select probability type:",
            ["Equal probabilities", "Custom probabilities"], 0, False)
        if not ok:
            return

        try:
            # Get observed frequencies
            observed = pd.to_numeric(self.data[observed_col], errors='coerce').dropna()
            n_categories = len(observed)

            if n_categories < 2:
                QMessageBox.warning(self, "Warning", "Need at least 2 categories for chi-square test")
                return

            # Get expected probabilities
            if prob_type == "Equal probabilities":
                expected_probs = np.ones(n_categories) / n_categories
            else:
                # Get custom probabilities from user
                probs = []
                total_prob = 0
                for i in range(n_categories):
                    prob, ok = QInputDialog.getDouble(self, f"Category {i+1}",
                        f"Enter probability for category {i+1}:",
                        1/n_categories, 0, 1, 4)
                    if not ok:
                        return
                    probs.append(prob)
                    total_prob += prob

                if abs(total_prob - 1) > 0.0001:
                    QMessageBox.warning(self, "Warning", "Probabilities must sum to 1")
                    return
                expected_probs = np.array(probs)

            # Calculate expected frequencies
            total_obs = sum(observed)
            expected = expected_probs * total_obs

            # Perform chi-square test
            chi2_stat, p_value = stats.chisquare(observed, expected)

            # Calculate contribution to chi-square for each category
            contributions = (observed - expected) ** 2 / expected

            # Create report
            report = f"""Chi-Square Goodness of Fit Test Results

Test Information:
Number of categories: {n_categories}
Total observations: {total_obs}

Test Statistics:
Chi-square statistic: {chi2_stat:.4f}
Degrees of freedom: {n_categories - 1}
p-value: {p_value:.4f}

Category Details:
{"Category":<10} {"Observed":<10} {"Expected":<10} {"Contribution":<10}
{"-"*40}
"""
            for i in range(n_categories):
                report += f"{i+1:<10} {observed[i]:<10.0f} {expected[i]:<10.2f} {contributions[i]:<10.4f}\n"

            report += f"""
Decision:
{"Reject" if p_value < 0.05 else "Fail to reject"} the null hypothesis at α = 0.05

Note: The null hypothesis is that the observed frequencies follow the specified distribution."""

            self.sessionWindow.setText(report)

            # Create visualization
            plt.figure(figsize=(10, 6))
            x = range(n_categories)
            width = 0.35

            plt.bar([i - width/2 for i in x], observed, width, label='Observed', color='blue', alpha=0.7)
            plt.bar([i + width/2 for i in x], expected, width, label='Expected', color='red', alpha=0.7)

            plt.xlabel('Category')
            plt.ylabel('Frequency')
            plt.title('Chi-Square Goodness of Fit Test\nObserved vs Expected Frequencies')
            plt.legend()
            plt.xticks(x)
            plt.grid(True, alpha=0.3)
            plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Error",
                f"An error occurred during chi-square analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def chiSquareIndependence(self):
        """Perform Chi-Square Test of Independence"""
        # Get first variable column
        var1_col = self.selectColumnDialog()
        if not var1_col:
            return

        # Get second variable column
        var2_col = self.selectColumnDialog()
        if not var2_col:
            return

        try:
            # Create contingency table
            contingency = pd.crosstab(self.data[var1_col], self.data[var2_col])
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            # Calculate contributions to chi-square
            contributions = (contingency - expected) ** 2 / expected

            # Create report
            report = f"""Chi-Square Test of Independence Results

Test Information:
Variables: {var1_col} and {var2_col}
Number of rows: {contingency.shape[0]}
Number of columns: {contingency.shape[1]}

Test Statistics:
Chi-square statistic: {chi2_stat:.4f}
Degrees of freedom: {dof}
p-value: {p_value:.4f}

Contingency Table:
{contingency.to_string()}

Expected Frequencies:
{pd.DataFrame(expected, index=contingency.index, columns=contingency.columns).to_string()}

Chi-Square Contributions:
{contributions.to_string()}

Decision:
{"Reject" if p_value < 0.05 else "Fail to reject"} the null hypothesis at α = 0.05

Note: The null hypothesis is that the variables are independent."""

            self.sessionWindow.setText(report)

            # Create visualization
            plt.figure(figsize=(12, 5))
            
            # Observed frequencies plot
            plt.subplot(1, 2, 1)
            sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd')
            plt.title('Observed Frequencies')
            
            # Expected frequencies plot
            plt.subplot(1, 2, 2)
            sns.heatmap(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns),
                       annot=True, fmt='.1f', cmap='YlOrRd')
            plt.title('Expected Frequencies')
            
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Error",
                f"An error occurred during chi-square analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def chiSquareHomogeneity(self):
        """Perform Chi-Square Test of Homogeneity"""
        # Get category column
        category_col = self.selectColumnDialog()
        if not category_col:
            return

        # Get group column
        group_col = self.selectColumnDialog()
        if not group_col:
            return

        try:
            # Create contingency table
            contingency = pd.crosstab(self.data[group_col], self.data[category_col])
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            # Calculate contributions to chi-square
            contributions = (contingency - expected) ** 2 / expected

            # Calculate proportions for each group
            proportions = contingency.div(contingency.sum(axis=1), axis=0)

            # Create report
            report = f"""Chi-Square Test of Homogeneity Results

Test Information:
Groups: {group_col}
Categories: {category_col}
Number of groups: {contingency.shape[0]}
Number of categories: {contingency.shape[1]}

Test Statistics:
Chi-square statistic: {chi2_stat:.4f}
Degrees of freedom: {dof}
p-value: {p_value:.4f}

Contingency Table (Frequencies):
{contingency.to_string()}

Proportions within Groups:
{proportions.to_string()}

Expected Frequencies:
{pd.DataFrame(expected, index=contingency.index, columns=contingency.columns).to_string()}

Chi-Square Contributions:
{contributions.to_string()}

Decision:
{"Reject" if p_value < 0.05 else "Fail to reject"} the null hypothesis at α = 0.05

Note: The null hypothesis is that the proportions are the same across groups."""

            self.sessionWindow.setText(report)

            # Create visualizations
            plt.figure(figsize=(15, 5))
            
            # Frequencies plot
            plt.subplot(1, 3, 1)
            sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd')
            plt.title('Observed Frequencies')
            
            # Proportions plot
            plt.subplot(1, 3, 2)
            sns.heatmap(proportions, annot=True, fmt='.2f', cmap='YlOrRd')
            plt.title('Proportions within Groups')
            
            # Expected frequencies plot
            plt.subplot(1, 3, 3)
            sns.heatmap(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns),
                       annot=True, fmt='.1f', cmap='YlOrRd')
            plt.title('Expected Frequencies')
            
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Error",
                f"An error occurred during chi-square analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MinitabLikeApp()
    mainWin.show()
    sys.exit(app.exec())
