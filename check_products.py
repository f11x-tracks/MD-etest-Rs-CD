import pandas as pd

df_cd = pd.read_csv('data/CD-data-sql.txt')
df_rs = pd.read_csv('data/filtered_ETestData.txt')

print('RBS Products:')
print(df_rs['PRODUCT'].unique())
print()
print('CD Products:')
print(df_cd['PRODUCT'].unique())
print()

# Check if any products match
rs_products = set(df_rs['PRODUCT'].unique())
cd_products = set(df_cd['PRODUCT'].unique())
common_products = rs_products & cd_products
print(f'Common products: {list(common_products)}')

# Check the first few wafers to see if there might be a mismatch in product naming
print('\nSample wafer data:')
common_wafers = list(set(df_cd['WAFERID']) & set(df_rs['WAFER_ID']))
if common_wafers:
    wafer = common_wafers[0]
    cd_wafer = df_cd[df_cd['WAFERID'] == wafer]
    rs_wafer = df_rs[df_rs['WAFER_ID'] == wafer]
    print(f'Wafer {wafer}:')
    if not cd_wafer.empty:
        print(f'  CD Product: {cd_wafer["PRODUCT"].iloc[0]}')
    if not rs_wafer.empty:
        print(f'  RS Product: {rs_wafer["PRODUCT"].iloc[0]}')

# Check coordinate ranges for the first common wafer
print(f'\nCoordinate analysis for wafer {wafer}:')
if not cd_wafer.empty and not rs_wafer.empty:
    print(f'  CD: X({cd_wafer["X"].min():.1f} to {cd_wafer["X"].max():.1f}), Y({cd_wafer["Y"].min():.1f} to {cd_wafer["Y"].max():.1f})')
    print(f'  RS: X({rs_wafer["X"].min():.1f} to {rs_wafer["X"].max():.1f}), Y({rs_wafer["Y"].min():.1f} to {rs_wafer["Y"].max():.1f})')
