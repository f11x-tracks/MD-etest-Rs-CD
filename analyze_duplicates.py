import pandas as pd
import numpy as np

# Load the current data
df = pd.read_csv('data/CD-data-sql.txt')
df['ENTITY_DATA_COLLECT_DATE'] = pd.to_datetime(df['ENTITY_DATA_COLLECT_DATE'])

print(f'Total rows: {len(df)}')
print(f'Date range: {df["ENTITY_DATA_COLLECT_DATE"].min()} to {df["ENTITY_DATA_COLLECT_DATE"].max()}')

# Check for duplicates
duplicate_cols = ['LAYER', 'WAFERID', 'CD_NAME', 'X', 'Y']
duplicates = df.duplicated(subset=duplicate_cols, keep=False)

if duplicates.any():
    print(f'\nFound {duplicates.sum()} duplicate rows')
    
    # Show a specific duplicate group
    dup_data = df[duplicates]
    first_group = dup_data.groupby(duplicate_cols).first().iloc[0:1]
    if not first_group.empty:
        first_key = first_group.index[0]
        group_data = df[(df['LAYER'] == first_key[0]) & 
                       (df['WAFERID'] == first_key[1]) & 
                       (df['CD_NAME'] == first_key[2]) &
                       (df['X'] == first_key[3]) &
                       (df['Y'] == first_key[4])]
        
        print(f'\nExample duplicate group:')
        print(f'Location: LAYER={first_key[0]}, WAFER={first_key[1]}, X={first_key[3]}, Y={first_key[4]}')
        group_sorted = group_data.sort_values('ENTITY_DATA_COLLECT_DATE')
        for _, row in group_sorted.iterrows():
            print(f'  {row["ENTITY_DATA_COLLECT_DATE"]}: CD={row["CD"]:.3f}')
            
    # Count duplicate groups
    unique_groups = dup_data.groupby(duplicate_cols).size()
    print(f'\nNumber of duplicate groups: {len(unique_groups)}')
    print(f'Group sizes: {unique_groups.value_counts().sort_index()}')
    
    # Show what would happen with keep='last' strategy
    print(f'\nTesting duplicate removal strategy:')
    df_test = df.sort_values(duplicate_cols + ['ENTITY_DATA_COLLECT_DATE']).drop_duplicates(
        subset=duplicate_cols, keep='last'
    )
    print(f'After keeping most recent: {len(df_test)} rows (removed {len(df) - len(df_test)})')
    
else:
    print('No duplicates found')
