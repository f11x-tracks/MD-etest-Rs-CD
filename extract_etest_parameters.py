#!/usr/bin/env python3
"""
Script to extract ETest parameter column names from filtered_ETestData.txt
and create a config_etest_parameters.txt file.

This script reads the filtered_ETestData.txt file and identifies all columns
that come after the PROGRAM column, which are considered parameter columns.
"""

import pandas as pd
import os

def extract_etest_parameters(input_file, output_file=None):
    """
    Extract parameter column names from filtered ETest data.
    
    Args:
        input_file (str): Path to the filtered_ETestData.txt file
        output_file (str): Path to the output config file. If None, will use 'config_etest_parameters.txt'
    """
    
    print(f"Reading data from: {input_file}")
    
    # Read the CSV file to get column names
    try:
        df = pd.read_csv(input_file, nrows=1)  # Only read header
        all_columns = df.columns.tolist()
        print(f"All columns found: {all_columns}")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Find the index of the PROGRAM column
    try:
        program_index = all_columns.index('PROGRAM')
        print(f"PROGRAM column found at index: {program_index}")
    except ValueError:
        print("Error: PROGRAM column not found in the data")
        return
    
    # Extract parameter columns (all columns after PROGRAM)
    parameter_columns = all_columns[program_index + 1:]
    
    print(f"\nParameter columns found ({len(parameter_columns)}):")
    for i, col in enumerate(parameter_columns, 1):
        print(f"  {i}. {col}")
    
    # Set output filename if not provided
    if output_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, 'config_etest_parameters.txt')
    
    # Save the parameter column names to config file
    try:
        with open(output_file, 'w') as f:
            f.write("# ETest Parameter Column Names\n")
            f.write("# Generated from filtered_ETestData.txt\n")
            f.write("# These are all columns that appear after the PROGRAM column\n\n")
            
            for col in parameter_columns:
                f.write(f"{col}\n")
        
        print(f"\nParameter column names saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving config file: {e}")

def main():
    """Main function to run the parameter extraction process."""
    
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'data', 'filtered_ETestData.txt')
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Please ensure filtered_ETestData.txt is in the data/ directory.")
        return
    
    # Run the parameter extraction
    extract_etest_parameters(input_file)
    
    print("\nParameter extraction completed successfully!")

if __name__ == "__main__":
    main()
