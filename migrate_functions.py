#!/usr/bin/env python3
"""
Migration Tool for Minitab-Like App

This script helps extract various functions from the monolithic app
and place them in the appropriate modular files.
"""

import re
import os

# Define output file structure
OUTPUT_STRUCTURE = {
    'stats/basic_stats.py': [
        'calculateDescriptiveStats',
        'calculateCorrelation',
    ],
    'stats/advanced_stats.py': [
        'hypothesisTesting',
        'performANOVA',
        'chiSquareTests',
        'regressionAnalysis',
    ],
    'stats/doe.py': [
        'createDOE',
        'analyzeDOE',
    ],
    'quality/control_charts.py': [
        'calculate_control_limits',
        'xbarRChart',
        'individualChart',
        'movingRangeChart',
    ],
    'quality/capability.py': [
        'calculate_capability_indices',
        'processCapability',
        'probabilityAnalysis',
    ],
    'quality/msa.py': [
        'gageRR',
        'linearityStudy',
        'biasStudy',
        'stabilityStudy',
    ],
    'sixsigma/dmaic.py': [
        'create_pareto_chart',
        'paretoChart',
        'fishboneDiagram',
        'fmeaTemplate',
    ],
    'sixsigma/metrics.py': [
        'calculate_dpmo',
        'dpmo_to_sigma',
        'dpmoCalculator',
        'sigmaLevelCalc',
        'yieldAnalysis',
    ],
    'calc/random_data.py': [
        'generateRandomData',
        'poissonDistribution',
    ],
}

# Path to monolithic app
MONOLITHIC_APP = 'minitab_like_app.py'
OUTPUT_BASE_DIR = 'minitab_app/modules'

def extract_function(source_code, function_name):
    """
    Extract a function from source code
    
    Args:
        source_code (str): Source code to search in
        function_name (str): Name of the function to extract
        
    Returns:
        str: Extracted function code or None if not found
    """
    # Look for function definition in class
    class_method_pattern = r'def\s+' + re.escape(function_name) + r'\s*\([^)]*\):\s*(?:"""(?:.*?)""")?.*?(?=\n\s*def|\n*$)'
    class_match = re.search(class_method_pattern, source_code, re.DOTALL)
    
    if class_match:
        return class_match.group(0)
    
    # Look for function definition outside class
    function_pattern = r'def\s+' + re.escape(function_name) + r'\s*\([^)]*\):.*?(?=\n(?:def|class)|\n*$)'
    function_match = re.search(function_pattern, source_code, re.DOTALL)
    
    if function_match:
        return function_match.group(0)
    
    return None

def create_module_file(module_path, functions, source_code):
    """
    Create a module file with extracted functions
    
    Args:
        module_path (str): Path to the module file
        functions (list): List of function names to extract
        source_code (str): Source code to extract from
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.join(OUTPUT_BASE_DIR, module_path)), exist_ok=True)
    
    # Get module name from path
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    
    # Start with a module docstring
    content = f'"""\n{module_name.replace("_", " ").title()} module\n"""\n\n'
    
    # Add imports
    content += 'import numpy as np\n'
    content += 'import pandas as pd\n'
    content += 'import matplotlib.pyplot as plt\n'
    
    # Extract functions
    for function_name in functions:
        function_code = extract_function(source_code, function_name)
        if function_code:
            content += '\n\n' + function_code
    
    # Write to file
    with open(os.path.join(OUTPUT_BASE_DIR, module_path), 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Main function to migrate code"""
    # Read monolithic app
    try:
        with open(MONOLITHIC_APP, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find {MONOLITHIC_APP}")
        return
    
    # Create module files
    for module_path, functions in OUTPUT_STRUCTURE.items():
        create_module_file(module_path, functions, source_code)
        print(f"Created {os.path.join(OUTPUT_BASE_DIR, module_path)}")

if __name__ == "__main__":
    main() 