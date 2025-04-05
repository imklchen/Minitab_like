"""
Basic Stats module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QCheckBox, QInputDialog, 
                           QDialogButtonBox, QMessageBox)


def descriptive_stats(data, col):
    """
    计算描述性统计
    
    Args:
        data (pd.DataFrame): 数据
        col (str): 要分析的列名
        
    Returns:
        tuple: (stats_dict, data_array) - 统计结果字典和数据数组
    """
    try:
        # 转换为数值类型，丢弃非数值
        data_array = pd.to_numeric(data[col], errors='coerce').dropna()
        
        # 计算统计量
        stats_dict = {
            'Count': len(data_array),
            'Mean': np.mean(data_array),
            'StDev': np.std(data_array, ddof=1),
            'Minimum': np.min(data_array),
            'Q1': np.percentile(data_array, 25),
            'Median': np.median(data_array),
            'Q3': np.percentile(data_array, 75),
            'Maximum': np.max(data_array)
        }
        
        return stats_dict, data_array
    except Exception as e:
        raise Exception(f"计算统计量错误: {str(e)}")


def correlation_analysis(data, col1, col2):
    """
    计算相关性
    
    Args:
        data (pd.DataFrame): 数据
        col1 (str): 第一列名
        col2 (str): 第二列名
        
    Returns:
        tuple: (corr_coef, p_value) - 相关系数和p值
    """
    try:
        # 转换为数值类型
        data1 = pd.to_numeric(data[col1], errors='coerce')
        data2 = pd.to_numeric(data[col2], errors='coerce')
        
        # 移除缺失值
        valid_mask = ~(pd.isna(data1) | pd.isna(data2))
        data1 = data1[valid_mask]
        data2 = data2[valid_mask]
        
        # 计算皮尔逊相关系数和p值
        corr_coef, p_value = stats.pearsonr(data1, data2)
        
        return corr_coef, p_value, data1, data2
    except Exception as e:
        raise Exception(f"计算相关性错误: {str(e)}")


def format_descriptive_stats(stats_dict, col):
    """
    格式化描述性统计结果
    
    Args:
        stats_dict (dict): 统计结果字典
        col (str): 列名
    
    Returns:
        str: 格式化后的统计结果
    """
    output = f"\nColumn: {col}\n"
    output += f"Count = {stats_dict['Count']}\n"
    output += f"Mean = {stats_dict['Mean']:.2f}\n"
    output += f"StDev = {stats_dict['StDev']:.2f}\n"
    output += f"Minimum = {stats_dict['Minimum']:.2f}\n"
    output += f"Q1 = {stats_dict['Q1']:.2f}\n"
    output += f"Median = {stats_dict['Median']:.2f}\n"
    output += f"Q3 = {stats_dict['Q3']:.2f}\n"
    output += f"Maximum = {stats_dict['Maximum']:.2f}"
    
    return output


def format_correlation_results(col1, col2, corr_coef, p_value, show_p_values=True):
    """
    格式化相关性分析结果
    
    Args:
        col1 (str): 第一列名
        col2 (str): 第二列名
        corr_coef (float): 相关系数
        p_value (float): p值
        show_p_values (bool): 是否显示p值
        
    Returns:
        str: 格式化后的结果
    """
    output = "Correlation Matrix:\n"
    output += f"            {col1}    {col2}\n"
    output += f"{col1}      1.000    {corr_coef:.3f}\n"
    output += f"{col2}      {corr_coef:.3f}    1.000"
    
    if show_p_values:
        output += "\n\nP-Values:\n"
        output += f"            {col1}    {col2}\n"
        output += f"{col1}        ---    {p_value:.3f}\n"
        output += f"{col2}      {p_value:.3f}      ---"
    
    return output


def create_descriptive_plots(data_array, col):
    """
    创建描述性统计图
    
    Args:
        data_array (np.array): 数据数组
        col (str): 列名
        
    Returns:
        matplotlib.figure.Figure: 图形对象
    """
    fig = plt.figure(figsize=(12, 5))
    
    # 直方图
    plt.subplot(1, 2, 1)
    plt.hist(data_array, bins='auto', density=False, alpha=0.7, color='skyblue')
    plt.title('Histogram')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    
    # 箱线图
    plt.subplot(1, 2, 2)
    plt.boxplot(data_array, labels=[col])
    plt.title('Box Plot')
    
    plt.tight_layout()
    return fig


def create_correlation_plot(data1, data2, col1, col2, corr_coef):
    """
    创建相关性分析图
    
    Args:
        data1 (np.array): 第一列数据
        data2 (np.array): 第二列数据
        col1 (str): 第一列名
        col2 (str): 第二列名
        corr_coef (float): 相关系数
        
    Returns:
        matplotlib.figure.Figure: 图形对象
    """
    fig = plt.figure(figsize=(10, 8))
    
    plt.scatter(data1, data2, alpha=0.7)
    plt.title(f'Correlation: {corr_coef:.3f}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    
    # 添加趋势线
    z = np.polyfit(data1, data2, 1)
    p = np.poly1d(z)
    plt.plot(data1, p(data1), "r--")
    
    plt.grid(alpha=0.3)
    return fig


# 原始方法保留为包装器，但将实际功能委托给上面的函数
def calculateDescriptiveStats(self):
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        col, ok = QInputDialog.getItem(self, "Select Column", 
            "Choose column for analysis:", self.data.columns.tolist(), 0, False)
        
        if ok and col:
            try:
                # 调用模块函数
                stats_dict, data_array = descriptive_stats(self.data, col)
                
                # 显示结果
                self.sessionWindow.clear()
                self.sessionWindow.append(format_descriptive_stats(stats_dict, col))
                
                # 创建可视化
                create_descriptive_plots(data_array, col)
                plt.show()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error calculating statistics: {str(e)}")

def calculateCorrelation(self):
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        # Get list of numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, "Warning", "At least two numeric columns are required")
            return

        # Column selection dialog
        col1, ok1 = QInputDialog.getItem(self, "Select First Column", 
            "Choose first column:", numeric_cols, 0, False)
        if not ok1:
            return

        col2, ok2 = QInputDialog.getItem(self, "Select Second Column", 
            "Choose second column:", numeric_cols, 0, False)
        if not ok2:
            return

        # Dialog options
        dialog = QDialog(self)
        dialog.setWindowTitle("Correlation Options")
        layout = QVBoxLayout()
        
        # Correlation type
        pearson_check = QCheckBox("Pearson correlation")
        pearson_check.setChecked(True)
        layout.addWidget(pearson_check)
        
        # P-values option
        pvalues_check = QCheckBox("Display p-values")
        pvalues_check.setChecked(True)
        layout.addWidget(pvalues_check)
        
        # OK and Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                # 调用模块函数
                corr_coef, p_value, data1, data2 = correlation_analysis(self.data, col1, col2)
                
                # 显示结果
                self.sessionWindow.clear()
                self.sessionWindow.append(
                    format_correlation_results(col1, col2, corr_coef, p_value, pvalues_check.isChecked())
                )
                
                # 创建相关性图
                create_correlation_plot(data1, data2, col1, col2, corr_coef)
                plt.show()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error calculating correlation: {str(e)}")