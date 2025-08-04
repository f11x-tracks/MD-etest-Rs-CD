import pandas as pd
import numpy as np

# Load data
df_cd = pd.read_csv('data/CD-data-sql.txt')
df_rs = pd.read_csv('data/filtered_ETestData.txt')

# Apply the same transformation that the app uses
def transform_rbs_coordinates(df_rs, df_cd):
    df_rs_transformed = df_rs.copy()
    common_wafers = list(set(df_cd['WAFERID']) & set(df_rs['WAFER_ID']))
    
    if not common_wafers:
        return df_rs_transformed
    
    all_cd_data = df_cd[df_cd['WAFERID'].isin(common_wafers)]
    all_rs_data = df_rs[df_rs['WAFER_ID'].isin(common_wafers)]
    
    rs_x_min, rs_x_max = all_rs_data['X'].min(), all_rs_data['X'].max()
    rs_y_min, rs_y_max = all_rs_data['Y'].min(), all_rs_data['Y'].max()
    cd_x_min, cd_x_max = all_cd_data['X'].min(), all_cd_data['X'].max()
    cd_y_min, cd_y_max = all_cd_data['Y'].min(), all_cd_data['Y'].max()
    
    x_offset = cd_x_min - rs_x_min
    y_offset = cd_y_min - rs_y_min
    
    df_rs_transformed['X'] = df_rs_transformed['X'] + x_offset
    df_rs_transformed['Y'] = df_rs_transformed['Y'] + y_offset
    
    return df_rs_transformed

df_rs = transform_rbs_coordinates(df_rs, df_cd)

# Filter for the specific case
target_lot = 'W519R19B'
target_layer = 'MET'

print(f'Investigating Lot={target_lot}, Layer={target_layer}')
print('=' * 60)

# Check CD data for this lot and layer
cd_lot_layer = df_cd[(df_cd['LOT'] == target_lot) & (df_cd['LAYER'] == target_layer)]
print(f'CD data for {target_lot}/{target_layer}: {len(cd_lot_layer)} rows')

if not cd_lot_layer.empty:
    measurement_sets = cd_lot_layer['MEASUREMENT_SET_NAME'].unique()
    print(f'  MEASUREMENT_SET_NAME values: {list(measurement_sets)}')
    fccd_data = cd_lot_layer[cd_lot_layer['MEASUREMENT_SET_NAME'].str.contains('FCCD', na=False)]
    print(f'  FCCD data: {len(fccd_data)} rows')
    if not fccd_data.empty:
        print(f'  FCCD wafers: {list(fccd_data["WAFERID"].unique())}')
        print(f'  FCCD CD_SITE values: {list(fccd_data["CD_SITE"].unique())}')

# Check RBS data for this lot
rs_lot = df_rs[df_rs['LOT'] == target_lot]
print(f'\nRBS data for {target_lot}: {len(rs_lot)} rows')
if not rs_lot.empty:
    print(f'  RBS wafers: {list(rs_lot["WAFER_ID"].unique())}')

# Find common wafers between CD and RBS for this lot
if not cd_lot_layer.empty and not rs_lot.empty:
    cd_wafers = set(cd_lot_layer['WAFERID'])
    rs_wafers = set(rs_lot['WAFER_ID'])
    common_wafers = cd_wafers & rs_wafers
    print(f'\nCommon wafers: {list(common_wafers)}')
    
    # Check what the dropdown filtering would show
    print(f'\nDropdown Analysis:')
    print(f'Available LAYER values: {list(df_cd["LAYER"].unique())}')
    met_data = df_cd[df_cd['LAYER'] == 'MET']
    print(f'CD_SITE values for MET layer: {list(met_data["CD_SITE"].unique())}')
    
    # Simulate the scatter plot filtering logic
    print(f'\nScatter Plot Filtering Simulation:')
    # The app uses selected_layer and selected_cd_site from dropdowns
    # Let's assume NEST_SP is selected (since that's the only CD_SITE)
    selected_layer = 'MET'
    selected_cd_site = 'NEST_SP'
    
    for wafer in common_wafers:
        print(f'\nWafer {wafer}:')
        
        # This is the logic from the app for FCCD scatter data
        fccd_data_wafer = cd_lot_layer[(cd_lot_layer['WAFERID'] == wafer) & 
                                       (cd_lot_layer['LAYER'] == target_layer) &
                                       (cd_lot_layer['MEASUREMENT_SET_NAME'].str.contains('FCCD', na=False))]
        
        print(f'  FCCD data for wafer: {len(fccd_data_wafer)} rows')
        
        # Apply scatter plot filtering
        fccd_scatter_data = fccd_data_wafer[
            (fccd_data_wafer['LAYER'] == selected_layer) & 
            (fccd_data_wafer['CD_SITE'] == selected_cd_site)
        ]
        
        print(f'  FCCD scatter data after filtering: {len(fccd_scatter_data)} rows')
        
        if not fccd_scatter_data.empty:
            print(f'    CD_SITE values: {list(fccd_scatter_data["CD_SITE"].unique())}')
            print(f'    LAYER values: {list(fccd_scatter_data["LAYER"].unique())}')
        else:
            print(f'    Checking why filtering failed:')
            if not fccd_data_wafer.empty:
                print(f'      Available LAYER values: {list(fccd_data_wafer["LAYER"].unique())}')
                print(f'      Available CD_SITE values: {list(fccd_data_wafer["CD_SITE"].unique())}')
                print(f'      Looking for LAYER={selected_layer}, CD_SITE={selected_cd_site}')
else:
    print('\nNo common wafers found!')
