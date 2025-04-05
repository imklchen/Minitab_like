#!/usr/bin/env python3
"""
Run script for the Minitab-like application using PyQt5 compatibility mode

This script directly imports the required PyQt5 components and runs the application
without any import hooks.
"""

import sys
import os
import re
import traceback

def modify_file_imports(file_path):
    """
    Temporarily modify PyQt6 imports to PyQt5 in the given file
    """
    if not os.path.isfile(file_path):
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Save backup if not already exists
    backup_path = file_path + '.bak'
    if not os.path.isfile(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # Replace imports
    modified = re.sub(r'from PyQt6\.(.*?) import', r'from PyQt5.\1 import', content)
    modified = re.sub(r'import PyQt6\.', r'import PyQt5.', modified)
    
    # Also ensure QAction is imported from QtWidgets in PyQt5
    modified = re.sub(r'from PyQt5\.QtGui import (.*?)QAction(.*?)', r'from PyQt5.QtGui import \1\2', modified)
    modified = re.sub(r'from PyQt5\.QtGui import QIcon,\s*', r'from PyQt5.QtGui import QIcon\n', modified)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified)
    
    return backup_path

def restore_file(file_path, backup_path):
    """
    Restore the original file from backup
    """
    if not (os.path.isfile(file_path) and os.path.isfile(backup_path)):
        return
    
    with open(backup_path, 'r', encoding='utf-8') as f:
        original = f.read()
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(original)

def modify_project_imports():
    """
    Modify PyQt imports in all Python files in the project
    """
    backups = {}
    modules_dir = os.path.join('minitab_app')
    
    for root, _, files in os.walk(modules_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                backup = modify_file_imports(file_path)
                if backup:
                    backups[file_path] = backup
    
    return backups

def restore_project_imports(backups):
    """
    Restore original imports from backups
    """
    for file_path, backup_path in backups.items():
        restore_file(file_path, backup_path)

if __name__ == "__main__":
    print("Converting PyQt6 imports to PyQt5...")
    backups = modify_project_imports()
    
    try:
        print("Running Minitab-like Application with PyQt5...")
        # Make sure PyQt5 is installed
        try:
            import PyQt5
            print(f"PyQt5 version: {PyQt5.QtCore.QT_VERSION_STR}")
        except ImportError:
            print("ERROR: PyQt5 is not installed. Please install it with 'pip install PyQt5'")
            sys.exit(1)
        
        try:
            from minitab_app.main import main
            sys.exit(main())
        except Exception as e:
            print(f"ERROR: Failed to run the application: {e}")
            print("Detailed error information:")
            traceback.print_exc()
            sys.exit(1)
    finally:
        print("Restoring original imports...")
        restore_project_imports(backups) 