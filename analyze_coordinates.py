import pandas as pd
import numpy as np

# Load data
df_cd = pd.read_csv('data/CD-data-sql.txt')
df_rs = pd.read_csv('data/filtered_ETestData.txt')

# Separate FCCD and DCCD
df_fccd = df_cd[df_cd['MEASUREMENT_SET_NAME'].str.contains('FCCD', na=False)]
df_dccd = df_cd[df_cd['MEASUREMENT_SET_NAME'].str.contains('DCCD', na=False)]

print('=== RBS Coordinate Analysis ===')
print(f'RBS X range: {df_rs["X"].min():.2f} to {df_rs["X"].max():.2f}')
print(f'RBS Y range: {df_rs["Y"].min():.2f} to {df_rs["Y"].max():.2f}')
print(f'RBS center (0,0) offset: X_center=0, Y_center=0')

print('\n=== FCCD Coordinate Analysis ===')
if not df_fccd.empty:
    print(f'FCCD X range: {df_fccd["X"].min():.2f} to {df_fccd["X"].max():.2f}')
    print(f'FCCD Y range: {df_fccd["Y"].min():.2f} to {df_fccd["Y"].max():.2f}')

print('\n=== DCCD Coordinate Analysis ===')
if not df_dccd.empty:
    print(f'DCCD X range: {df_dccd["X"].min():.2f} to {df_dccd["X"].max():.2f}')
    print(f'DCCD Y range: {df_dccd["Y"].min():.2f} to {df_dccd["Y"].max():.2f}')

# Check by product
print('\n=== By Product Analysis ===')
products = df_rs['PRODUCT'].unique()
for product in products[:3]:  # Show first 3 products
    rs_prod = df_rs[df_rs['PRODUCT'] == product]
    cd_prod = df_cd[df_cd['PRODUCT'] == product]
    
    print(f'\nProduct: {product}')
    print(f'  RBS: X({rs_prod["X"].min():.1f} to {rs_prod["X"].max():.1f}), Y({rs_prod["Y"].min():.1f} to {rs_prod["Y"].max():.1f})')
    if not cd_prod.empty:
        print(f'  CD:  X({cd_prod["X"].min():.1f} to {cd_prod["X"].max():.1f}), Y({cd_prod["Y"].min():.1f} to {cd_prod["Y"].max():.1f})')

# Show sample coordinates to understand the pattern
print('\n=== Sample Coordinates ===')
print('RBS sample (first 5 points):')
print(df_rs[['X', 'Y']].head())

print('\nFCCD sample (first 5 points):')
if not df_fccd.empty:
    print(df_fccd[['X', 'Y']].head())

print('\nDCCD sample (first 5 points):')
if not df_dccd.empty:
    print(df_dccd[['X', 'Y']].head())
