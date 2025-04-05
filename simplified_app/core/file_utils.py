"""
File utilities for loading and saving data
"""

import os
import pandas as pd

def load_data(file_path):
    """
    Load data from a file into a pandas DataFrame
    
    Args:
        file_path (str): Path to the file to load
        
    Returns:
        pandas.DataFrame: Loaded data
        
    Raises:
        Exception: If file cannot be loaded
    """
    try:
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.csv':
            return pd.read_csv(file_path)
        elif ext.lower() in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        elif ext.lower() == '.txt':
            # Try to auto-detect separator
            return pd.read_csv(file_path, sep=None, engine='python')
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        raise Exception(f"Error loading file: {e}")

def save_data(data, file_path):
    """
    Save data to a file
    
    Args:
        data (pandas.DataFrame): Data to save
        file_path (str): Path to save the file to
        
    Raises:
        Exception: If file cannot be saved
    """
    try:
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.csv':
            data.to_csv(file_path, index=False)
        elif ext.lower() in ['.xls', '.xlsx']:
            data.to_excel(file_path, index=False)
        elif ext.lower() == '.txt':
            data.to_csv(file_path, sep='\t', index=False)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        raise Exception(f"Error saving file: {e}")
