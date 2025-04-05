import pandas as pd
import os
from PyQt6.QtWidgets import QTableWidgetItem

def load_data(file_path):
    """加载数据文件"""
    try:
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.csv':
            return pd.read_csv(file_path)
        elif ext.lower() in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        elif ext.lower() == '.txt':
            return pd.read_csv(file_path, sep=None, engine='python')
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    except Exception as e:
        raise Exception(f"加载文件错误: {e}")

def save_data(data, file_path):
    """保存数据到文件"""
    try:
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.csv':
            data.to_csv(file_path, index=False)
        elif ext.lower() in ['.xls', '.xlsx']:
            data.to_excel(file_path, index=False)
        elif ext.lower() == '.txt':
            data.to_csv(file_path, sep='\t', index=False)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    except Exception as e:
        raise Exception(f"保存文件错误: {e}")

def data_to_table(data, table_widget):
    """
    Update QTableWidget with data from DataFrame
    
    Args:
        data (pandas.DataFrame): Data to display
        table_widget (QTableWidget): Table widget to update
    """
    # Clear table
    table_widget.clearContents()
    
    # Set table dimensions
    nrows, ncols = data.shape
    table_widget.setRowCount(nrows)
    table_widget.setColumnCount(ncols)
    
    # Set headers
    table_widget.setHorizontalHeaderLabels(data.columns)
    
    # Fill data
    for i in range(nrows):
        for j in range(ncols):
            value = data.iloc[i, j]
            if pd.isna(value):
                value = ""
            table_widget.setItem(i, j, QTableWidgetItem(str(value)))

def table_to_data(table_widget):
    """
    Create DataFrame from QTableWidget data
    
    Args:
        table_widget (QTableWidget): Table widget to get data from
        
    Returns:
        pandas.DataFrame: Data from table
    """
    rows = table_widget.rowCount()
    cols = table_widget.columnCount()
    
    # Get column names
    headers = [table_widget.horizontalHeaderItem(i).text() for i in range(cols)]
    
    # Create empty DataFrame
    data = pd.DataFrame(columns=headers)
    
    # Fill DataFrame with values
    for i in range(rows):
        row_data = []
        for j in range(cols):
            item = table_widget.item(i, j)
            if item is not None and item.text().strip():
                row_data.append(item.text())
            else:
                row_data.append(None)
        
        if any(x is not None for x in row_data):  # Only add non-empty rows
            data.loc[len(data)] = row_data
    
    # Convert numeric columns
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='ignore')
    
    return data
