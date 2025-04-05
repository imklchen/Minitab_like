"""
Control Charts module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QComboBox, 
                           QSpinBox, QDialogButtonBox, QMessageBox,
                           QLabel, QInputDialog, QCheckBox, QHBoxLayout, QLineEdit)
from minitab_app.core.constants import CONTROL_CHART_CONSTANTS, MR_CONTROL_CHART_CONSTANTS


def calculate_control_limits(data, n=5):
    """
    计算 X-bar 和 R 控制图的控制限
    
    Args:
        data: 形状为 (m, n) 的 numpy 数组，其中 m 是子组数，n 是子组大小
        n: 子组大小 (默认 5)
        
    Returns:
        dict: 包含控制限和其他统计量的字典
    """
    # 根据子组大小获取常量
    if n in CONTROL_CHART_CONSTANTS:
        A2 = CONTROL_CHART_CONSTANTS[n]['A2']
        D3 = CONTROL_CHART_CONSTANTS[n]['D3']
        D4 = CONTROL_CHART_CONSTANTS[n]['D4']
    else:
        # 默认常量 (n=5)
        A2 = 0.577
        D3 = 0
        D4 = 2.115
    
    # 计算统计量
    xbar = np.mean(data, axis=1)  # 每个子组的平均值
    ranges = np.ptp(data, axis=1)  # 每个子组的极差
    
    overall_mean = np.mean(xbar)
    mean_range = np.mean(ranges)
    
    # 计算控制限
    ucl_x = overall_mean + A2 * mean_range
    lcl_x = overall_mean - A2 * mean_range
    ucl_r = D4 * mean_range
    lcl_r = D3 * mean_range
    
    # 对于 n=5，系数为 2.326
    d2 = 2.326 if n == 5 else 2
    std_dev = mean_range / d2
    
    return {
        'center_x': overall_mean,
        'ucl_x': ucl_x,
        'lcl_x': lcl_x,
        'center_r': mean_range,
        'ucl_r': ucl_r,
        'lcl_r': lcl_r,
        'std_dev': std_dev,
        'xbar': xbar,
        'ranges': ranges
    }


def calculate_mr_control_limits(data, mr_length=2):
    """
    计算个体控制图和移动极差控制图的控制限
    
    Args:
        data: 数据数组
        mr_length: 移动极差长度 (默认 2)
        
    Returns:
        tuple: (i_limits, mr_limits) - 个体和移动极差控制限
    """
    # 计算移动极差
    moving_ranges = np.zeros(len(data) - mr_length + 1)
    for i in range(len(moving_ranges)):
        moving_ranges[i] = np.ptp(data[i:i+mr_length])
    
    # 计算均值和移动极差均值
    mean = np.mean(data)
    mr_mean = np.mean(moving_ranges)
    
    # 根据 MR 长度获取常量
    if mr_length in MR_CONTROL_CHART_CONSTANTS:
        E2 = MR_CONTROL_CHART_CONSTANTS[mr_length]['E2']
        D3 = MR_CONTROL_CHART_CONSTANTS[mr_length]['D3']
        D4 = MR_CONTROL_CHART_CONSTANTS[mr_length]['D4']
    else:
        # 默认常量 (n=2)
        E2 = 2.66
        D3 = 0
        D4 = 3.267
    
    # 个体控制限
    i_ucl = mean + E2 * mr_mean
    i_lcl = mean - E2 * mr_mean
    
    # 移动极差控制限
    mr_ucl = D4 * mr_mean
    mr_lcl = D3 * mr_mean
    
    i_limits = {
        'data': data,
        'center': mean,
        'ucl': i_ucl,
        'lcl': i_lcl
    }
    
    mr_limits = {
        'data': moving_ranges,
        'center': mr_mean,
        'ucl': mr_ucl,
        'lcl': mr_lcl
    }
    
    return i_limits, mr_limits


def create_xbar_r_charts(limits):
    """
    创建 X-bar 和 R 控制图
    
    Args:
        limits: 控制限字典
        
    Returns:
        matplotlib.figure.Figure: 图形对象
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # X-bar 图
    ax1.plot(range(1, len(limits['xbar']) + 1), limits['xbar'], marker='o', color='blue')
    ax1.axhline(y=limits['center_x'], color='g', linestyle='-', label='Center')
    ax1.axhline(y=limits['ucl_x'], color='r', linestyle='--', label='UCL')
    ax1.axhline(y=limits['lcl_x'], color='r', linestyle='--', label='LCL')
    ax1.set_title('X-bar Chart')
    ax1.set_xlabel('Subgroup')
    ax1.set_ylabel('Sample Mean')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # R 图
    ax2.plot(range(1, len(limits['ranges']) + 1), limits['ranges'], marker='o', color='blue')
    ax2.axhline(y=limits['center_r'], color='g', linestyle='-', label='Center')
    ax2.axhline(y=limits['ucl_r'], color='r', linestyle='--', label='UCL')
    ax2.axhline(y=limits['lcl_r'], color='r', linestyle='--', label='LCL')
    ax2.set_title('R Chart')
    ax2.set_xlabel('Subgroup')
    ax2.set_ylabel('Range')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig


def create_individual_mr_charts(i_limits, mr_limits):
    """
    创建个体和移动极差控制图
    
    Args:
        i_limits: 个体控制限字典
        mr_limits: 移动极差控制限字典
        
    Returns:
        matplotlib.figure.Figure: 图形对象
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 个体图
    ax1.plot(range(1, len(i_limits['data']) + 1), i_limits['data'], marker='o', color='blue')
    ax1.axhline(y=i_limits['center'], color='g', linestyle='-', label='CL')
    ax1.axhline(y=i_limits['ucl'], color='r', linestyle='--', label='UCL')
    ax1.axhline(y=i_limits['lcl'], color='r', linestyle='--', label='LCL')
    ax1.set_title('Individual Chart')
    ax1.set_xlabel('Observation')
    ax1.set_ylabel('Individual Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 移动极差图
    ax2.plot(range(1, len(mr_limits['data']) + 1), mr_limits['data'], marker='o', color='blue')
    ax2.axhline(y=mr_limits['center'], color='g', linestyle='-', label='CL')
    ax2.axhline(y=mr_limits['ucl'], color='r', linestyle='--', label='UCL')
    ax2.axhline(y=mr_limits['lcl'], color='r', linestyle='--', label='LCL')
    ax2.set_title('Moving Range Chart')
    ax2.set_xlabel('Observation')
    ax2.set_ylabel('Moving Range')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig


def format_xbar_r_results(limits, subgroup_size, num_subgroups):
    """
    格式化 X-bar 和 R 控制图结果
    
    Args:
        limits: 控制限字典
        subgroup_size: 子组大小
        num_subgroups: 子组数量
        
    Returns:
        str: 格式化的结果文本
    """
    result_text = "X-bar R Chart Analysis Results\n\n"
    result_text += "Test Information:\n"
    result_text += f"Number of subgroups: {num_subgroups}\n"
    result_text += f"Subgroup size: {subgroup_size}\n"
    result_text += f"Total observations: {num_subgroups * subgroup_size}\n\n"
    
    result_text += "Control Limits:\n"
    result_text += "X-bar Chart:\n"
    result_text += f"- Center Line (CL): {limits['center_x']:.4f}\n"
    result_text += f"- Upper Control Limit (UCL): {limits['ucl_x']:.4f}\n"
    result_text += f"- Lower Control Limit (LCL): {limits['lcl_x']:.4f}\n\n"
    
    result_text += "R Chart:\n"
    result_text += f"- Center Line (CL): {limits['center_r']:.4f}\n"
    result_text += f"- Upper Control Limit (UCL): {limits['ucl_r']:.4f}\n"
    result_text += f"- Lower Control Limit (LCL): {limits['lcl_r']:.4f}\n\n"
    
    result_text += "Process Statistics:\n"
    result_text += f"- Overall Mean: {limits['center_x']:.4f}\n"
    result_text += f"- Average Range: {limits['center_r']:.4f}\n"
    result_text += f"- Standard Deviation: {limits['std_dev']:.4f}\n"
    
    return result_text


def format_individual_mr_results(i_limits, mr_limits, col_name):
    """
    格式化个体和移动极差控制图结果
    
    Args:
        i_limits: 个体控制限字典
        mr_limits: 移动极差控制限字典
        col_name: 列名
        
    Returns:
        str: 格式化的结果文本
    """
    points_out_i = sum((i_limits['data'] > i_limits['ucl']) | (i_limits['data'] < i_limits['lcl']))
    points_out_mr = sum((mr_limits['data'] > mr_limits['ucl']) | (mr_limits['data'] < mr_limits['lcl']))
    
    result_text = f"Individual Chart Analysis for {col_name}\n\n"
    result_text += "Individual Chart Statistics:\n"
    result_text += f"Mean: {i_limits['center']:.3f}\n"
    result_text += f"UCL: {i_limits['ucl']:.3f}\n"
    result_text += f"LCL: {i_limits['lcl']:.3f}\n\n"
    result_text += f"Number of Points: {len(i_limits['data'])}\n"
    result_text += f"Points Outside Control Limits: {points_out_i}\n\n"
    
    result_text += "Moving Range Chart Statistics:\n"
    result_text += f"Mean MR: {mr_limits['center']:.3f}\n"
    result_text += f"UCL: {mr_limits['ucl']:.3f}\n"
    result_text += f"LCL: {mr_limits['lcl']:.3f}\n"
    result_text += f"Number of Ranges: {len(mr_limits['data'])}\n"
    result_text += f"Ranges Outside Control Limits: {points_out_mr}\n\n"
    
    result_text += f"Note: Control limits are based on ±3 sigma\n"
    
    return result_text


# 以下是原有的类方法，现在作为包装器

def xbarRChart(self):
        """Create X-bar and R control charts"""
        try:
            # Get list of numeric columns only
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 5:
                QMessageBox.warning(self, "Warning", "Need at least 5 numeric columns for samples")
                return

            # Prompt for each sample selection
            sample_cols = []
            prompts = [
                "Select the first variable (Sample1)",
                "Select the second variable (Sample2)",
                "Select the third variable (Sample3)",
                "Select the fourth variable (Sample4)",
                "Select the fifth variable (Sample5)"
            ]
            
            for prompt in prompts:
                col = self.selectColumnDialog(prompt, numeric_cols)  # Pass numeric_cols instead of all columns
                if not col:
                    return  # User cancelled
                sample_cols.append(col)
            
            # Convert data to numeric and check for missing values
            n_rows = len(self.data)
            data = []
            for i in range(n_rows):
                row_data = []
                for col in sample_cols:
                    try:
                        val = float(self.data.loc[i, col])
                        if np.isnan(val):
                            raise ValueError
                        row_data.append(val)
                    except (ValueError, TypeError):
                        QMessageBox.warning(self, "Warning", f"Invalid value in row {i+1}, column {col}")
                        return
                data.append(row_data)
            
            data = np.array(data)
            
            # 调用模块函数
            limits = calculate_control_limits(data, n=5)
            
            # 创建图表
            fig = create_xbar_r_charts(limits)
            plt.show()
            
            # 格式化结果
            result_text = format_xbar_r_results(limits, 5, len(data))
            self.sessionWindow.setText(result_text)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")
            return

def individualChart(self):
        """Create individual control chart"""
        try:
            # Get list of numeric columns only
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 1:
                QMessageBox.warning(self, "Warning", "Need at least one numeric column for measurements")
                return

            # Create dialog for options
            dialog = QDialog(self)
            dialog.setWindowTitle("Individual Chart Options")
            layout = QVBoxLayout()

            # Column selection
            col_label = QLabel("Select Measurement column:")
            layout.addWidget(col_label)
            
            col_combo = QComboBox()
            col_combo.addItems(numeric_cols)
            layout.addWidget(col_combo)

            # Display Tests option
            display_tests = QCheckBox("Display Tests")
            layout.addWidget(display_tests)

            # Alpha value selection
            alpha_layout = QHBoxLayout()
            alpha_label = QLabel("α value for control limits:")
            alpha_input = QLineEdit("0.05")  # Default value
            alpha_layout.addWidget(alpha_label)
            alpha_layout.addWidget(alpha_input)
            layout.addLayout(alpha_layout)

            # Moving Range length selection
            mr_layout = QHBoxLayout()
            mr_label = QLabel("Moving Range length:")
            mr_input = QLineEdit("2")  # Default value
            mr_layout.addWidget(mr_label)
            mr_layout.addWidget(mr_input)
            layout.addLayout(mr_layout)

            # Add OK and Cancel buttons
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | 
                QDialogButtonBox.StandardButton.Cancel
            )
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            dialog.setLayout(layout)
            
            # Show dialog and get results
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            # Get selected options
            col = col_combo.currentText()
            show_tests = display_tests.isChecked()
            try:
                alpha = float(alpha_input.text())
                if not 0 < alpha < 1:
                    raise ValueError("Alpha must be between 0 and 1")
            except ValueError:
                QMessageBox.warning(self, "Warning", "Invalid α value. Using default 0.05")
                alpha = 0.05

            try:
                mr_length = int(mr_input.text())
                if mr_length < 2:
                    raise ValueError("Moving Range length must be at least 2")
            except ValueError:
                QMessageBox.warning(self, "Warning", "Invalid Moving Range length. Using default 2")
                mr_length = 2

            # Get the data and check for missing values
            data = pd.to_numeric(self.data[col], errors='coerce')
            if data.isna().any():
                QMessageBox.warning(self, "Warning", "Missing values found in the selected column")
                return
                
            # 调用模块函数
            i_limits, mr_limits = calculate_mr_control_limits(data.values, mr_length)
            
            # 创建图表
            fig = create_individual_mr_charts(i_limits, mr_limits)
            plt.show()
            
            # 格式化结果
            result_text = format_individual_mr_results(i_limits, mr_limits, col)
            
            # 添加测试信息（如选择了显示测试）
            if show_tests:
                result_text += "\nTests for Special Causes:\n"
                # 基于选择的 alpha 值添加测试结果
            
            self.sessionWindow.setText(result_text)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")
            return

def movingRangeChart(self, col=None):
        """Create moving range chart"""
        if col is None:
            col = self.selectColumnDialog()
            if not col:
                return
                
        try:
            # Get numeric data
            data = pd.to_numeric(self.data[col], errors='coerce').dropna()
            
            if len(data) < 2:
                QMessageBox.warning(self, "Warning", "Insufficient data for moving range chart")
                return
                
            # 调用模块函数
            i_limits, mr_limits = calculate_mr_control_limits(data.values)
            
            # 创建图表
            fig = create_individual_mr_charts(i_limits, mr_limits)
            plt.show()
            
            # 格式化结果
            result_text = format_individual_mr_results(i_limits, mr_limits, col)
            self.sessionWindow.setText(result_text)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")