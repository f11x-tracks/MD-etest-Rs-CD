#!/usr/bin/env python3
"""
Script to filter ETestData.txt and keep only columns that begin with specified prefixes.

This script reads the ETestData.txt file and filters it to keep only columns that start with:
- LOT
- WAFER
- X
- Y
- TEST_END_DATE (note: corrected from TESET_END_DATE)
- WAFER_ID
- OPERATION
- PRODUCT
- PROGRAM
- RBS_MFW2
- RBS_MF2W2
"""

import pandas as pd
import os

def filter_etest_data(input_file, output_file=None):
    """
    Filter ETestData.txt to keep only columns with specified prefixes.
    
    Args:
        input_file (str): Path to the input ETestData.txt file
        output_file (str): Path to the output filtered file. If None, will use 'filtered_ETestData.txt'
    """
    
    # Define the column prefixes to keep
    keep_prefixes = [
        'LOT',
        'WAFER', 
        'X',
        'Y',
        'TEST_END_DATE',  # Note: corrected from TESET_END_DATE in user request
        'WAFER_ID',
        'OPERATION',
        'PRODUCT', 
        'PROGRAM',
        'RBS_MFW2',
        'RBS_MF2W2'
    ]
    
    print(f"Reading data from: {input_file}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        print(f"Total columns: {len(df.columns)}")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Get column names and filter them
    all_columns = df.columns.tolist()
    
    # Find columns that start with any of the specified prefixes
    columns_to_keep = []
    for col in all_columns:
        for prefix in keep_prefixes:
            if col.startswith(prefix):
                columns_to_keep.append(col)
                break  # Break to avoid adding the same column multiple times
    
    print(f"\nColumns to keep ({len(columns_to_keep)}):")
    for i, col in enumerate(columns_to_keep, 1):
        print(f"  {i:2d}. {col}")
    
    # Filter the dataframe
    df_filtered = df[columns_to_keep]
    print(f"\nFiltered data shape: {df_filtered.shape}")
    
    # Clean up column names by removing [PROBE]@ETEST or @ETEST suffixes
    print(f"\nCleaning column names...")
    original_columns = df_filtered.columns.tolist()
    cleaned_columns = []
    
    for col in original_columns:
        # Remove [PROBE]@ETEST suffix first, then @ETEST suffix
        cleaned_col = col
        if cleaned_col.endswith('[PROBE]@ETEST'):
            cleaned_col = cleaned_col[:-len('[PROBE]@ETEST')]
        elif cleaned_col.endswith('@ETEST'):
            cleaned_col = cleaned_col[:-len('@ETEST')]
        cleaned_columns.append(cleaned_col)
    
    # Rename the columns
    df_filtered.columns = cleaned_columns
    
    # Show the column name changes
    print(f"Column name changes:")
    for orig, clean in zip(original_columns, cleaned_columns):
        if orig != clean:
            print(f"  '{orig}' -> '{clean}'")
    
    # Set output filename if not provided
    if output_file is None:
        input_dir = os.path.dirname(input_file)
        output_file = os.path.join(input_dir, 'filtered_ETestData.txt')
    
    # Save the filtered data
    try:
        df_filtered.to_csv(output_file, index=False)
        print(f"\nFiltered data saved to: {output_file}")
        
        # Show some sample data
        print(f"\nFirst 5 rows of filtered data:")
        print(df_filtered.head())
        
        # Show data types and non-null counts for each column
        print(f"\nColumn information:")
        print(df_filtered.info())
        
    except Exception as e:
        print(f"Error saving file: {e}")

def main():
    """Main function to run the filtering process."""
    
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'ETestData.txt')
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Please ensure ETestData.txt is in the same directory as this script.")
        return
    
    # Run the filtering
    filter_etest_data(input_file)
    
    print("\nFiltering completed successfully!")

if __name__ == "__main__":
    main()
