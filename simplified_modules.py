"""
创建简化版本的模块化 Minitab-Like 应用程序
"""

import os
import sys

def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_file(path, content):
    """创建文件并写入内容"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_simplified_app():
    """创建简化版本的应用程序"""
    # 创建目录结构
    ensure_dir('simplified_app')
    ensure_dir('simplified_app/core')
    ensure_dir('simplified_app/ui')
    
    # 创建 __init__.py 文件
    create_file('simplified_app/__init__.py', '')
    create_file('simplified_app/core/__init__.py', '')
    create_file('simplified_app/ui/__init__.py', '')
    
    # 创建主入口文件
    main_py = '''"""
Minitab-Like Application - Simplified Version
Main entry point
"""

import sys
from simplified_app.core.app import MinitabApp

def main():
    """Main entry point function"""
    app = MinitabApp()
    return app.run()

if __name__ == "__main__":
    sys.exit(main())
'''
    create_file('simplified_app/main.py', main_py)
    
    # 创建核心应用程序类
    app_py = '''"""
Core application class
"""

import sys
import pandas as pd
from PyQt6.QtWidgets import QApplication

from simplified_app.ui.main_window import MinitabMainWindow

class MinitabApp:
    """
    Main application class that initializes and manages the application
    """
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.main_window = None
        self.data = pd.DataFrame()
        self.current_file = None
    
    def run(self):
        """Initialize and run the application"""
        self.main_window = MinitabMainWindow(self)
        self.main_window.show()
        return self.app.exec()
    
    def get_data(self):
        """Return the current data"""
        return self.data
    
    def set_data(self, data):
        """Set the application data"""
        self.data = data
        # Update UI if main window exists
        if self.main_window:
            self.main_window.update_table()
    
    def get_current_file(self):
        """Return the current file path"""
        return self.current_file
    
    def set_current_file(self, file_path):
        """Set the current file path"""
        self.current_file = file_path
        # Update window title if main window exists
        if self.main_window:
            self.main_window.update_title()
'''
    create_file('simplified_app/core/app.py', app_py)
    
    # 创建文件工具类
    file_utils_py = '''"""
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
            data.to_csv(file_path, sep='\\t', index=False)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        raise Exception(f"Error saving file: {e}")
'''
    create_file('simplified_app/core/file_utils.py', file_utils_py)
    
    # 创建主窗口类
    main_window_py = '''"""
Main application window
"""

import sys
import pandas as pd
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
                           QMenuBar, QTextEdit, QFileDialog, QMessageBox, QAction)
from PyQt6.QtCore import Qt

from simplified_app.core.file_utils import load_data, save_data

class MinitabMainWindow(QMainWindow):
    """
    Main application window - Simplified version
    """
    def __init__(self, app_controller):
        super().__init__()
        self.app_controller = app_controller
        self.setWindowTitle("Minitab-Like Tool (Simplified)")
        self.resize(900, 600)
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        # Create central widget with table and session window
        self.table = QTableWidget(50, 10)
        self.table.setHorizontalHeaderLabels([f"C{i+1}" for i in range(10)])
        
        self.sessionWindow = QTextEdit()
        self.sessionWindow.setReadOnly(True)
        
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.sessionWindow)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Create simplified menu bar
        self.createMenuBar()
    
    def createMenuBar(self):
        """Create a simplified application menu bar"""
        menuBar = QMenuBar()
        self.setMenuBar(menuBar)
        
        # Create main menus - just File menu for now
        fileMenu = menuBar.addMenu("File")
        
        # File Menu
        fileMenu.addAction(self.makeAction("Open", self.openFile))
        fileMenu.addAction(self.makeAction("Save", self.saveFile))
        fileMenu.addAction(self.makeAction("Save As", self.saveFileAs))
        fileMenu.addSeparator()
        fileMenu.addAction(self.makeAction("Clear Table Data", self.clearTableData))
        fileMenu.addSeparator()
        fileMenu.addAction(self.makeAction("Exit", self.close))
    
    def makeAction(self, name, func):
        """Create a QAction for menus"""
        action = QAction(name, self)
        action.triggered.connect(func)
        return action
    
    def loadDataFromTable(self):
        """Load data from the table into a DataFrame"""
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        
        # Get headers
        headers = [self.table.horizontalHeaderItem(i).text() for i in range(cols)]
        
        # Create empty DataFrame
        data = pd.DataFrame(columns=headers)
        
        # Fill data
        for i in range(rows):
            row_data = []
            has_data = False
            
            for j in range(cols):
                item = self.table.item(i, j)
                if item and item.text().strip():
                    row_data.append(item.text())
                    has_data = True
                else:
                    row_data.append(None)
            
            if has_data:
                data.loc[len(data)] = row_data
        
        # Convert numeric columns
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='ignore')
        
        return data
    
    def update_table(self):
        """Update the table with current data"""
        data = self.app_controller.get_data()
        
        # Clear table
        self.table.clearContents()
        
        # Set dimensions
        if not data.empty:
            nrows, ncols = data.shape
            self.table.setRowCount(max(nrows, 50))  # Minimum 50 rows
            self.table.setColumnCount(ncols)
            
            # Set headers
            self.table.setHorizontalHeaderLabels(data.columns)
            
            # Fill data
            for i in range(nrows):
                for j in range(ncols):
                    value = data.iloc[i, j]
                    if pd.isna(value):
                        value = ""
                    self.table.setItem(i, j, QTableWidgetItem(str(value)))
    
    def update_title(self):
        """Update window title with current file"""
        file_path = self.app_controller.get_current_file()
        if file_path:
            self.setWindowTitle(f"Minitab-Like Tool (Simplified) - {file_path}")
        else:
            self.setWindowTitle("Minitab-Like Tool (Simplified)")
    
    def log_message(self, message):
        """Add message to session window"""
        self.sessionWindow.append(message)
    
    # File operations
    def openFile(self):
        """Open a file dialog and load data"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                data = load_data(file_path)
                self.app_controller.set_data(data)
                self.app_controller.set_current_file(file_path)
                self.log_message(f"Loaded data from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
    
    def saveFile(self):
        """Save data to current file or open Save As dialog"""
        file_path = self.app_controller.get_current_file()
        if not file_path:
            return self.saveFileAs()
        
        try:
            data = self.loadDataFromTable()
            save_data(data, file_path)
            self.log_message(f"Saved data to {file_path}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return False
    
    def saveFileAs(self):
        """Open a Save As dialog and save data"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Data File", "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;Text Files (*.txt)"
        )
        
        if file_path:
            try:
                data = self.loadDataFromTable()
                save_data(data, file_path)
                self.app_controller.set_current_file(file_path)
                self.log_message(f"Saved data to {file_path}")
                return True
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                return False
        
        return False
    
    def clearTableData(self):
        """Clear all data from the table"""
        reply = QMessageBox.question(
            self, "Clear Data", 
            "Are you sure you want to clear all data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.app_controller.set_data(pd.DataFrame(columns=[f"C{i+1}" for i in range(10)]))
            self.app_controller.set_current_file(None)
            self.log_message("Cleared all data")
'''
    create_file('simplified_app/ui/main_window.py', main_window_py)
    
    # 创建启动脚本
    run_script = '''#!/usr/bin/env python3
"""
Simplified Minitab-Like Application Launcher
"""

import sys
from simplified_app.main import main

if __name__ == "__main__":
    sys.exit(main())
'''
    create_file('run_simplified.py', run_script)
    
    print("创建简化版本的模块化应用程序完成！")
    print("运行方式: python run_simplified.py")

if __name__ == "__main__":
    create_simplified_app() 