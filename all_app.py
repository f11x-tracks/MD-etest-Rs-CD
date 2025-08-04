import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

# Load the data
df_cd = pd.read_csv('data/CD-data-sql.txt')
df_rs = pd.read_csv('data/filtered_ETestData.txt')

# Separate FCCD and DCCD data
df_fccd = df_cd[df_cd['MEASUREMENT_SET_NAME'].str.contains('FCCD', na=False)].copy()
df_dccd = df_cd[df_cd['MEASUREMENT_SET_NAME'].str.contains('DCCD', na=False)].copy()

def transform_rbs_coordinates(df_rs, df_cd):
    """
    Transform RBS coordinates from center-based (0,0 at wafer center) 
    to corner-based (0,0 at lower-left) to match FCCD/DCCD coordinate system.
    
    Calculate transformation separately for each product to handle different coordinate systems.
    """
    df_rs_transformed = df_rs.copy()
    
    # Get common wafers to determine coordinate transformation
    common_wafers = list(set(df_cd['WAFERID']) & set(df_rs['WAFER_ID']))
    
    if not common_wafers:
        print("  No common wafers found for coordinate transformation")
        return df_rs_transformed
    
    print("Analyzing coordinate systems using common wafers by product:")
    
    # Get common data for analysis
    all_cd_data = df_cd[df_cd['WAFERID'].isin(common_wafers)]
    all_rs_data = df_rs[df_rs['WAFER_ID'].isin(common_wafers)]
    
    if all_cd_data.empty or all_rs_data.empty:
        print("  No data found for common wafers")
        return df_rs_transformed
    
    # Group by product to calculate separate transformations
    products = all_cd_data['PRODUCT'].unique()
    
    transformations = {}
    
    for product in products:
        print(f"\n  Product: {product}")
        
        # Get wafers for this product
        product_wafers = all_cd_data[all_cd_data['PRODUCT'] == product]['WAFERID'].unique()
        
        # Filter data for this product
        cd_product_data = all_cd_data[all_cd_data['PRODUCT'] == product]
        rs_product_data = all_rs_data[all_rs_data['WAFER_ID'].isin(product_wafers)]
        
        if cd_product_data.empty or rs_product_data.empty:
            print(f"    No matching data for product {product}")
            continue
        
        # Calculate coordinate ranges for this product
        rs_x_min, rs_x_max = rs_product_data['X'].min(), rs_product_data['X'].max()
        rs_y_min, rs_y_max = rs_product_data['Y'].min(), rs_product_data['Y'].max()
        cd_x_min, cd_x_max = cd_product_data['X'].min(), cd_product_data['X'].max()
        cd_y_min, cd_y_max = cd_product_data['Y'].min(), cd_product_data['Y'].max()
        
        # Calculate transformation for this product
        x_offset = cd_x_min - rs_x_min
        y_offset = cd_y_min - rs_y_min
        
        transformations[product] = {'x_offset': x_offset, 'y_offset': y_offset}
        
        print(f"    RBS original:    X({rs_x_min:.1f} to {rs_x_max:.1f}), Y({rs_y_min:.1f} to {rs_y_max:.1f})")
        print(f"    CD target:       X({cd_x_min:.1f} to {cd_x_max:.1f}), Y({cd_y_min:.1f} to {cd_y_max:.1f})")
        print(f"    Transformation:  X+{x_offset:.1f}, Y+{y_offset:.1f}")
    
    # Apply product-specific transformations to ALL RBS data
    print(f"\nApplying product-specific transformations to {len(df_rs_transformed)} RBS measurement points:")
    
    for product, transform in transformations.items():
        # Find all wafers for this product in the RBS data
        # We need to map RBS WAFER_ID to CD PRODUCT via the common wafer mapping
        product_wafer_ids = all_cd_data[all_cd_data['PRODUCT'] == product]['WAFERID'].unique()
        
        # Apply transformation to RBS data for wafers of this product
        mask = df_rs_transformed['WAFER_ID'].isin(product_wafer_ids)
        count = mask.sum()
        
        if count > 0:
            df_rs_transformed.loc[mask, 'X'] += transform['x_offset']
            df_rs_transformed.loc[mask, 'Y'] += transform['y_offset']
            print(f"  {product}: {count} points transformed by X+{transform['x_offset']:.1f}, Y+{transform['y_offset']:.1f}")
    
    return df_rs_transformed

print(f"Loaded CD data: {len(df_cd)} rows")
print(f"FCCD data: {len(df_fccd)} rows")
print(f"DCCD data: {len(df_dccd)} rows")
print(f"Loaded RS data: {len(df_rs)} rows")
print(f"CD data columns: {df_cd.columns.tolist()}")
print(f"RS data columns: {df_rs.columns.tolist()}")

# Transform RBS coordinates to match CD coordinate system BEFORE filtering
print("=== Coordinate System Transformation ===")
df_rs = transform_rbs_coordinates(df_rs, df_cd)
print("RBS coordinates transformed to match FCCD/DCCD coordinate system")

# Get unique LAYER and CD_SITE values for dropdowns
unique_layers = sorted(df_cd['LAYER'].unique())
layer_cd_site_map = {}
for layer in unique_layers:
    layer_cd_site_map[layer] = sorted(df_cd[df_cd['LAYER'] == layer]['CD_SITE'].unique())

print(f"Unique LAYER values: {unique_layers}")
print(f"CD_SITE by LAYER: {layer_cd_site_map}")

# Get common WAFER_IDs - matching WAFERID in CD data to WAFER_ID in RS data
cd_wafer_ids = set(df_cd['WAFERID'].unique())
rs_wafer_ids = set(df_rs['WAFER_ID'].unique())
common_wafer_ids = list(cd_wafer_ids & rs_wafer_ids)
print(f"Found {len(common_wafer_ids)} common WAFER_IDs")

# Filter data to only common wafers (using transformed RBS data)
df_fccd_common = df_fccd[df_fccd['WAFERID'].isin(common_wafer_ids)]
df_dccd_common = df_dccd[df_dccd['WAFERID'].isin(common_wafer_ids)]
df_rs_common = df_rs[df_rs['WAFER_ID'].isin(common_wafer_ids)]

# Group wafers by PRODUCT, LOT, and LAYER from CD data (using combined data for grouping)
df_cd_common = df_cd[df_cd['WAFERID'].isin(common_wafer_ids)]

# Create a mapping of WAFER_ID to LOT from RS data for proper grouping
wafer_to_rs_lot = dict(zip(df_rs_common['WAFER_ID'], df_rs_common['LOT']))

# Group wafers by PRODUCT from CD data and LAYER from CD data, but use RS LOT
# First get all combinations, then map to RS LOT
cd_groups = df_cd_common.groupby(['PRODUCT', 'LAYER'])['WAFERID'].unique().to_dict()

# Create new grouping with RS LOT values
wafer_groups = {}
for (product, layer), wafer_ids in cd_groups.items():
    # For each wafer group, get the RS LOT for proper grouping
    for wafer_id in wafer_ids:
        if wafer_id in wafer_to_rs_lot:
            rs_lot = wafer_to_rs_lot[wafer_id]
            key = (product, rs_lot, layer)
            if key not in wafer_groups:
                wafer_groups[key] = []
            wafer_groups[key].append(wafer_id)

# Convert lists back to arrays for consistency
wafer_groups = {k: np.array(v) for k, v in wafer_groups.items()}

print(f"Found {len(wafer_groups)} unique PRODUCT, LOT (from RS), and LAYER combinations")

def create_contour_plot(df, x_col, y_col, z_col, title, colorscale='Viridis'):
    """Create a contour plot from scattered data"""
    if df.empty:
        # Return empty plot if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=486)
        return fig
    
    # Get the data points
    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values
    
    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[mask], y[mask], z[mask]
    
    if len(x) == 0:
        # Return empty plot if no valid data
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=486)
        return fig
    
    # Create a grid for interpolation - stay within data bounds
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Only add minimal padding if we have a very small range
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # If data covers a very small range, add some minimum spacing
    if x_range == 0:
        x_min -= 0.5
        x_max += 0.5
    if y_range == 0:
        y_min -= 0.5
        y_max += 0.5
    
    # Create grid that stays within the actual data bounds
    grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    
    # Interpolate data to grid
    try:
        # Use linear interpolation only within the convex hull of data points
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
        
        # Don't use nearest neighbor to fill - this was causing extrapolation
        # Instead, let NaN areas remain as gaps in the contour
        
        # Check if we have any valid interpolated values
        if np.all(np.isnan(grid_z)):
            # If no interpolation possible, fall back to scatter plot
            raise ValueError("No interpolation possible")
        
        # Create contour plot
        fig = go.Figure()
        
        # Add contour - only where we have valid interpolated data
        fig.add_trace(go.Contour(
            x=grid_x[:, 0],
            y=grid_y[0, :],
            z=grid_z.T,  # Transpose the grid to fix orientation
            colorscale=colorscale,
            showscale=True,
            line=dict(width=0.5),
            contours=dict(
                coloring='fill',
                start=np.nanmin(grid_z),
                end=np.nanmax(grid_z),
                size=(np.nanmax(grid_z) - np.nanmin(grid_z)) / 15  # Fewer levels for cleaner look
            ),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z:.3f}<extra></extra>',
            connectgaps=False  # Don't connect across gaps - important!
        ))
        
        # Add scatter points to show actual data locations
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(
                size=3,  # Slightly smaller points
                color='white',
                line=dict(width=0.8, color='black')
            ),
            name='Data Points',
            hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{customdata:.3f}<extra></extra>',
            customdata=z
        ))
        
    except Exception as e:
        # If interpolation fails, create scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(
                size=8,
                color=z,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title="Value")
            ),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{marker.color:.3f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        height=486,  # Reduced by another 10% from 540 to 486
        width=486,   # Set width equal to height for square plots
        margin=dict(l=40, r=40, t=50, b=40),  # Original margins
        xaxis=dict(autorange=True),  # Let X-axis auto-scale to show all data
        yaxis=dict(autorange=True),  # Let Y-axis auto-scale to show all data
        font=dict(size=12)  # Original font size
    )
    
    return fig

# Create a function to generate plots based on filter criteria
def generate_plots(filter_option, show_summary, selected_layer=None, selected_cd_site=None):
    """Generate plots based on filter and display options"""
    plots = []
    plot_count = 0

    for (product, lot, layer), wafer_ids in wafer_groups.items():
        # Skip if layer is NaN
        if pd.isna(layer):
            continue
        
        # NEW: Only show groups that match the selected layer from dropdown
        if selected_layer and layer != selected_layer:
            continue
            
        # Determine which RBS column to use based on layer
        if layer == 'MDT':
            rbs_column = 'RBS_MFW2'
            rbs_title = 'RBS_MFW2 (Rs)'
        elif layer == 'MET':
            rbs_column = 'RBS_MF2W2'
            rbs_title = 'RBS_MF2W2 (Rs)'
        else:
            continue  # Skip unknown layers
        
        # Filter and collect valid wafers for this group
        valid_wafers = []
        
        for wafer_id in sorted(wafer_ids):
            # Filter FCCD data for selected wafer and layer
            fccd_data = df_fccd_common[(df_fccd_common['WAFERID'] == wafer_id) & 
                                     (df_fccd_common['LAYER'] == layer)]
            
            # Filter DCCD data for selected wafer and layer
            dccd_data = df_dccd_common[(df_dccd_common['WAFERID'] == wafer_id) & 
                                     (df_dccd_common['LAYER'] == layer)]
            
            # Apply additional filtering based on selected layer and CD_SITE for scatter plots
            if selected_layer and selected_cd_site:
                # For scatter plots, filter CD data by selected layer and CD_SITE
                fccd_scatter_data = fccd_data[
                    (fccd_data['LAYER'] == selected_layer) & 
                    (fccd_data['CD_SITE'] == selected_cd_site)
                ].copy()
                dccd_scatter_data = dccd_data[
                    (dccd_data['LAYER'] == selected_layer) & 
                    (dccd_data['CD_SITE'] == selected_cd_site)
                ].copy()
            else:
                # Use all available data if no specific layer/site selected
                fccd_scatter_data = fccd_data.copy()
                dccd_scatter_data = dccd_data.copy()
            
            # Filter RS data for selected wafer with valid values in the selected RBS column
            rs_data = df_rs_common[(df_rs_common['WAFER_ID'] == wafer_id) & 
                                 (df_rs_common[rbs_column].notna())].copy()
            
            # Skip if no RS data for this wafer
            if rs_data.empty:
                continue
                
            # Apply filter if selected
            if filter_option == 'filtered':
                if len(rs_data) <= 10:  # Check data points for the specific RBS measurement
                    continue
            
            # Check if we have at least one type of CD data
            if fccd_data.empty and dccd_data.empty:
                continue
            
            valid_wafers.append((wafer_id, rs_data, fccd_data, dccd_data, fccd_scatter_data, dccd_scatter_data))
        
        # Skip group if no valid wafers
        if not valid_wafers:
            continue
            
        # Limit to max 25 wafers
        valid_wafers = valid_wafers[:25]
        
        # Create section header for each product/lot/layer group (lot is already from RS data)
        group_header = html.Div([
            html.H2(f"Product: {product} | Lot: {lot} | Layer: {layer} ({len(valid_wafers)} wafers)", 
                   style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 20,
                          'backgroundColor': '#e9ecef', 'padding': '10px', 'borderRadius': '5px'})
        ])
        plots.append(group_header)
        
        # Create summary table for this group
        summary_data = []
        for wafer_id, rs_data, fccd_data, dccd_data, fccd_scatter_data, dccd_scatter_data in valid_wafers:
            # Calculate statistics for each measurement type
            rbs_avg = rs_data[rbs_column].mean() if not rs_data.empty else np.nan
            rbs_std = rs_data[rbs_column].std() if not rs_data.empty else np.nan
            
            fccd_avg = fccd_data['CD'].mean() if not fccd_data.empty else np.nan
            fccd_std = fccd_data['CD'].std() if not fccd_data.empty else np.nan
            
            dccd_avg = dccd_data['CD'].mean() if not dccd_data.empty else np.nan
            dccd_std = dccd_data['CD'].std() if not dccd_data.empty else np.nan
            
            # Add scatter data counts
            fccd_scatter_count = len(fccd_scatter_data) if not fccd_scatter_data.empty else 0
            dccd_scatter_count = len(dccd_scatter_data) if not dccd_scatter_data.empty else 0
            
            summary_data.append({
                'Wafer_ID': wafer_id,
                'RBS_Avg': f"{rbs_avg:.3f}" if not np.isnan(rbs_avg) else "N/A",
                'RBS_StdDev': f"{rbs_std:.3f}" if not np.isnan(rbs_std) else "N/A",
                'FCCD_Avg': f"{fccd_avg:.3f}" if not np.isnan(fccd_avg) else "N/A",
                'FCCD_StdDev': f"{fccd_std:.3f}" if not np.isnan(fccd_std) else "N/A",
                'DCCD_Avg': f"{dccd_avg:.3f}" if not np.isnan(dccd_avg) else "N/A",
                'DCCD_StdDev': f"{dccd_std:.3f}" if not np.isnan(dccd_std) else "N/A",
                'FCCD_Scatter_Count': fccd_scatter_count,
                'DCCD_Scatter_Count': dccd_scatter_count
            })
        
        # Create summary table component
        summary_table = html.Div([
            html.H3("Summary Statistics", style={'textAlign': 'center', 'marginBottom': 15, 'color': '#333'}),
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Wafer ID", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th(f"{rbs_title} Avg", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th(f"{rbs_title} StdDev", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th("FCCD Avg", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th("FCCD StdDev", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th("DCCD Avg", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th("DCCD StdDev", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th("FCCD Scatter Pts", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#e6f3ff'}),
                        html.Th("DCCD Scatter Pts", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#e6f3ff'})
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(row['Wafer_ID'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['RBS_Avg'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['RBS_StdDev'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['FCCD_Avg'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['FCCD_StdDev'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['DCCD_Avg'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['DCCD_StdDev'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['FCCD_Scatter_Count'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center', 'backgroundColor': '#f0f8ff'}),
                        html.Td(row['DCCD_Scatter_Count'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center', 'backgroundColor': '#f0f8ff'})
                    ]) for row in summary_data
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'margin': '0 auto'})
        ], style={
            'marginBottom': '30px', 
            'padding': '15px', 
            'backgroundColor': '#f8f9fa', 
            'borderRadius': '5px',
            'border': '1px solid #dee2e6'
        })
        
        plots.append(summary_table)
        
        # Create plots for each wafer in this group - each wafer on its own row
        for wafer_id, rs_data, fccd_data, dccd_data, fccd_scatter_data, dccd_scatter_data in valid_wafers:
            print(f"Creating plots for wafer: {wafer_id}")
                
            # Create RBS plot
            fig_rs = create_contour_plot(
                rs_data, 'X', 'Y', rbs_column, 
                f'{rbs_title} - {wafer_id}',
                colorscale='RdBu'  # Red to Blue (reversed for resistivity)
            )
            
            # Create FCCD plot if data exists
            fig_fccd = None
            if not fccd_data.empty:
                fig_fccd = create_contour_plot(
                    fccd_data, 'X', 'Y', 'CD', 
                    f'CD (FCCD) - {wafer_id}',
                    colorscale='RdBu_r'  # Blue to Red (normal for dimension)
                )
            
            # Create DCCD plot if data exists
            fig_dccd = None
            if not dccd_data.empty:
                fig_dccd = create_contour_plot(
                    dccd_data, 'X', 'Y', 'CD', 
                    f'CD (DCCD) - {wafer_id}',
                    colorscale='RdBu_r'  # Blue to Red (normal for dimension)
                )
            
            # Create layout based on available data
            plot_divs = []
            plot_count_for_wafer = 1  # Always have RBS
            
            # Add RBS plot
            plot_divs.append(
                html.Div([
                    dcc.Graph(figure=fig_rs, style={'height': '486px'})
                ], style={'width': '19%', 'display': 'inline-block', 'verticalAlign': 'top'})
            )
            
            # Add FCCD plot if available
            if fig_fccd is not None:
                plot_count_for_wafer += 1
                plot_divs.append(
                    html.Div([
                        dcc.Graph(figure=fig_fccd, style={'height': '486px'})
                    ], style={'width': '19%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1%'})
                )
            
            # Add DCCD plot if available
            if fig_dccd is not None:
                plot_count_for_wafer += 1
                plot_divs.append(
                    html.Div([
                        dcc.Graph(figure=fig_dccd, style={'height': '486px'})
                    ], style={'width': '19%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1%'})
                )
            
            # Add scatter plots to the same row (all with consistent 1% margin)
            # Use filtered scatter data based on selected layer and CD_SITE
            
            # FCCD vs RBS scatter plot
            if not fccd_scatter_data.empty and not rs_data.empty:
                # Use nearest neighbor matching instead of exact coordinate matching
                fccd_points = fccd_scatter_data[['X', 'Y', 'CD']].copy()
                rs_points = rs_data[['X', 'Y', rbs_column]].copy()
                
                # For each FCCD point, find the nearest RBS point
                merged_data = []
                for _, fccd_row in fccd_points.iterrows():
                    fccd_x, fccd_y, fccd_cd = fccd_row['X'], fccd_row['Y'], fccd_row['CD']
                    
                    # Calculate distances to all RBS points
                    distances = np.sqrt((rs_points['X'] - fccd_x)**2 + (rs_points['Y'] - fccd_y)**2)
                    
                    # Find closest RBS point (within reasonable distance)
                    min_dist = distances.min()
                    if min_dist <= 2.0:  # Allow up to 2 units distance for matching
                        closest_idx = distances.idxmin()
                        rbs_value = rs_points.loc[closest_idx, rbs_column]
                        merged_data.append({
                            'FCCD': fccd_cd,
                            'RBS': rbs_value,
                            'distance': min_dist
                        })
                
                if merged_data:
                    merged_fccd_rbs = pd.DataFrame(merged_data)
                    
                    fig_fccd_rbs = go.Figure()
                    fig_fccd_rbs.add_trace(go.Scatter(
                        x=merged_fccd_rbs['FCCD'],
                        y=merged_fccd_rbs['RBS'],
                        mode='markers',
                        marker=dict(
                            size=6, 
                            color='darkred',
                            opacity=0.7
                        ),
                        name='FCCD vs RBS',
                        hovertemplate='FCCD: %{x:.3f}<br>RBS: %{y:.3f}<br>Match distance: %{customdata:.2f}<extra></extra>',
                        customdata=merged_fccd_rbs['distance']
                    ))
                    fig_fccd_rbs.update_layout(
                        title=f'FCCD vs {rbs_title}<br><sub>Layer: {selected_layer}, Site: {selected_cd_site} ({len(merged_fccd_rbs)} matched points)</sub>',
                        xaxis_title='FCCD',
                        yaxis_title=rbs_title,
                        height=486,
                        width=486,
                        margin=dict(l=30, r=30, t=50, b=30),
                        font=dict(size=10)
                    )
                    plot_divs.append(
                        html.Div([
                            dcc.Graph(figure=fig_fccd_rbs, style={'height': '486px'})
                        ], style={'width': '19%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1%'})
                    )
            
            # DCCD vs RBS scatter plot
            if not dccd_scatter_data.empty and not rs_data.empty:
                # Use nearest neighbor matching instead of exact coordinate matching
                dccd_points = dccd_scatter_data[['X', 'Y', 'CD']].copy()
                rs_points = rs_data[['X', 'Y', rbs_column]].copy()
                
                # For each DCCD point, find the nearest RBS point
                merged_data = []
                for _, dccd_row in dccd_points.iterrows():
                    dccd_x, dccd_y, dccd_cd = dccd_row['X'], dccd_row['Y'], dccd_row['CD']
                    
                    # Calculate distances to all RBS points
                    distances = np.sqrt((rs_points['X'] - dccd_x)**2 + (rs_points['Y'] - dccd_y)**2)
                    
                    # Find closest RBS point (within reasonable distance)
                    min_dist = distances.min()
                    if min_dist <= 2.0:  # Allow up to 2 units distance for matching
                        closest_idx = distances.idxmin()
                        rbs_value = rs_points.loc[closest_idx, rbs_column]
                        merged_data.append({
                            'DCCD': dccd_cd,
                            'RBS': rbs_value,
                            'distance': min_dist
                        })
                
                if merged_data:
                    merged_dccd_rbs = pd.DataFrame(merged_data)
                    
                    fig_dccd_rbs = go.Figure()
                    fig_dccd_rbs.add_trace(go.Scatter(
                        x=merged_dccd_rbs['DCCD'],
                        y=merged_dccd_rbs['RBS'],
                        mode='markers',
                        marker=dict(
                            size=6, 
                            color='darkblue',
                            opacity=0.7
                        ),
                        name='DCCD vs RBS',
                        hovertemplate='DCCD: %{x:.3f}<br>RBS: %{y:.3f}<br>Match distance: %{customdata:.2f}<extra></extra>',
                        customdata=merged_dccd_rbs['distance']
                    ))
                    fig_dccd_rbs.update_layout(
                        title=f'DCCD vs {rbs_title}<br><sub>Layer: {selected_layer}, Site: {selected_cd_site} ({len(merged_dccd_rbs)} matched points)</sub>',
                        xaxis_title='DCCD',
                        yaxis_title=rbs_title,
                        height=486,
                        width=486,
                        margin=dict(l=30, r=30, t=50, b=30),
                        font=dict(size=10)
                    )
                    plot_divs.append(
                        html.Div([
                            dcc.Graph(figure=fig_dccd_rbs, style={'height': '540px'})
                        ], style={'width': '19%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1%'})
                    )
            
            # Create single row layout for this wafer
            wafer_plots = html.Div([
                html.H4(f"Wafer: {wafer_id}", style={'textAlign': 'center', 'marginBottom': 10})
            ] + plot_divs, style={'marginBottom': '20px'})
            
            # Add data summary if enabled
            summary_section = html.Div()
            if show_summary == 'show':
                summary_section = html.Div([
                    html.H4(f"Data Summary for {wafer_id}", style={'textAlign': 'center'}),
                    html.Div([
                        html.Div([
                            html.P(f"{rbs_title} Data: {len(rs_data)} points"),
                            html.P(f"Range: {rs_data[rbs_column].min():.3f} - {rs_data[rbs_column].max():.3f}" if not rs_data.empty else "No data"),
                        ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P(f"FCCD Data: {len(fccd_data)} points" if not fccd_data.empty else "FCCD: No data"),
                            html.P(f"Range: {fccd_data['CD'].min():.3f} - {fccd_data['CD'].max():.3f}" if not fccd_data.empty else "No data"),
                        ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center', 'marginLeft': '2%'}),
                        
                        html.Div([
                            html.P(f"DCCD Data: {len(dccd_data)} points" if not dccd_data.empty else "DCCD: No data"),
                            html.P(f"Range: {dccd_data['CD'].min():.3f} - {dccd_data['CD'].max():.3f}" if not dccd_data.empty else "No data"),
                        ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center', 'marginLeft': '2%'}),
                        
                        html.Div([
                            html.P(f"FCCD Scatter: {len(fccd_scatter_data)} points" if not fccd_scatter_data.empty else "FCCD Scatter: No data"),
                            html.P(f"Layer: {selected_layer}, Site: {selected_cd_site}"),
                        ], style={'width': '17%', 'display': 'inline-block', 'textAlign': 'center', 'marginLeft': '2%', 'backgroundColor': '#f0f8ff', 'padding': '5px', 'borderRadius': '3px'}),
                        
                        html.Div([
                            html.P(f"DCCD Scatter: {len(dccd_scatter_data)} points" if not dccd_scatter_data.empty else "DCCD Scatter: No data"),
                            html.P(f"Layer: {selected_layer}, Site: {selected_cd_site}"),
                        ], style={'width': '17%', 'display': 'inline-block', 'textAlign': 'center', 'marginLeft': '2%', 'backgroundColor': '#f0f8ff', 'padding': '5px', 'borderRadius': '3px'})
                    ])
                ], style={'marginTop': 10, 'marginBottom': 20, 'padding': 10, 'backgroundColor': '#f8f9fa', 'borderRadius': 5})
            
            # Combine wafer plots and summary
            wafer_content = html.Div([
                wafer_plots,
                summary_section
            ], style={'marginBottom': '30px', 'border': '1px solid #dee2e6', 'borderRadius': '5px', 'padding': '15px'})
            
            plots.append(wafer_content)
        
        plot_count += len(valid_wafers)
    
    return plots, plot_count

# Initialize the Dash app
app = dash.Dash(__name__)

# Add callback to update CD_SITE dropdown based on LAYER selection
@app.callback(
    Output('cd-site-dropdown', 'options'),
    Output('cd-site-dropdown', 'value'),
    [Input('layer-dropdown', 'value')]
)
def update_cd_site_options(selected_layer):
    if selected_layer is None:
        return [], None
    
    cd_sites = layer_cd_site_map.get(selected_layer, [])
    options = [{'label': site, 'value': site} for site in cd_sites]
    value = cd_sites[0] if cd_sites else None
    
    return options, value

# Add callback for dynamic plot generation
@app.callback(
    Output('plots-container', 'children'),
    [Input('filter-radio', 'value'),
     Input('summary-radio', 'value'),
     Input('layer-dropdown', 'value'),
     Input('cd-site-dropdown', 'value')]
)
def update_plots(filter_option, show_summary, selected_layer, selected_cd_site):
    plots, plot_count = generate_plots(filter_option, show_summary, selected_layer, selected_cd_site)
    
    if not plots:
        return html.Div([
            html.H3(f"No wafers match the current filter criteria (Layer: {selected_layer})", 
                   style={'textAlign': 'center', 'marginTop': 50, 'color': 'gray'})
        ])
    
    # Add summary at the top
    summary_header = html.Div([
        html.H3(f"Displaying {plot_count} wafers for Layer: {selected_layer}, CD Site: {selected_cd_site}", 
               style={'textAlign': 'center', 'marginBottom': 20, 'color': '#333'})
    ])
    
    return [summary_header] + plots

# Generate all plots at startup - this is now replaced by the callback
# print("Generating all plots...")
# all_plots = []
# plot_count = 0

# for (product, lot), wafer_ids in wafer_groups.items():
#     # Create section header for each product/lot group
#     group_header = html.Div([
#         html.H2(f"Product: {product} | Lot: {lot}", 
#                style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 20,
#                       'backgroundColor': '#e9ecef', 'padding': '10px', 'borderRadius': '5px'})
#     ])
#     all_plots.append(group_header)
#     
#     # Create plots for each wafer in this group
#     for wafer_id in sorted(wafer_ids):
#         # Filter data for selected wafer
#         rs_data = df_rs[df_rs['WAFER_ID'] == wafer_id]
#         cd_data = df_cd[df_cd['WAFER_ID'] == wafer_id]
#         
#         # Skip if no CD data for this wafer
#         if cd_data.empty:
#             continue
#         
#         plot_count += 1
#         print(f"Creating plots for wafer {plot_count}: {wafer_id}")
#             
#         # Create plots with inverse color scales to show correlation
#         fig_rs = create_contour_plot(
#             rs_data, 'X', 'Y', 'RBS_MFW2', 
#             f'RBS_MFW2 (Rs) - {wafer_id}',
#             colorscale='RdBu'  # Red to Blue (reversed for resistivity)
#         )
#         
#         fig_cd = create_contour_plot(
#             cd_data, 'X', 'Y', 'CD', 
#             f'CD (FCCD) - {wafer_id}',
#             colorscale='RdBu_r'  # Blue to Red (normal for dimension)
#         )
#         
#         # Create side-by-side layout for this wafer
#         wafer_plots = html.Div([
#             html.Div([
#                 dcc.Graph(figure=fig_rs, style={'height': '400px'})
#             ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
#             
#             html.Div([
#                 dcc.Graph(figure=fig_cd, style={'height': '400px'})
#             ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
#             
#             # Data summary for this wafer
#             html.Div([
#                 html.H4(f"Data Summary for {wafer_id}", style={'textAlign': 'center'}),
#                 html.Div([
#                     html.Div([
#                         html.P(f"RBS_MFW2 Data: {len(rs_data)} points"),
#                         html.P(f"Range: {rs_data['RBS_MFW2'].min():.3f} - {rs_data['RBS_MFW2'].max():.3f}" if not rs_data.empty else "No data"),
#                     ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center'}),
#                     
#                     html.Div([
#                         html.P(f"CD Data: {len(cd_data)} points"),
#                         html.P(f"Range: {cd_data['CD'].min():.3f} - {cd_data['CD'].max():.3f}" if not cd_data.empty else "No data"),
#                     ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center', 'marginLeft': '10%'})
#                 ])
#             ], style={'marginTop': 10, 'marginBottom': 20, 'padding': 10, 'backgroundColor': '#f8f9fa', 'borderRadius': 5})
#         ], style={'marginBottom': '30px', 'border': '1px solid #dee2e6', 'borderRadius': '5px', 'padding': '15px'})
#         
#         all_plots.append(wafer_plots)

# print(f"Generated {plot_count} wafer plot pairs")

# Define the layout
app.layout = html.Div([
    html.H1("All Wafer Contour Plot Comparison", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    html.Div([
        html.Div([
            html.P("Color Scale: Red = High CD (large FCCD/DCCD) / Low RBS (low Rs)", 
                   style={'color': 'red', 'fontWeight': 'bold', 'margin': '5px 0'}),
            html.P("Blue = Low CD (small FCCD/DCCD) / High RBS (high Rs)", 
                   style={'color': 'blue', 'fontWeight': 'bold', 'margin': '5px 0'}),
            html.P("Correlated areas should show similar colors. MDT layers use RBS_MFW2, MET layers use RBS_MF2W2. Shows RBS, FCCD, and DCCD when available.", 
                   style={'fontStyle': 'italic', 'fontSize': 14, 'margin': '5px 0'}),
            html.P("All plots and tables are filtered by the selected Layer and CD Site. Only Product/Lot/Layer groups matching the dropdown selections are shown.", 
                   style={'fontStyle': 'italic', 'fontSize': 12, 'margin': '5px 0', 'color': '#666'})
        ], style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'})
    ], style={'textAlign': 'center', 'marginBottom': 20}),
    
    # Control options
    html.Div([
        html.Div([
            html.Label("Filter Options:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
            dcc.RadioItems(
                id='filter-radio',
                options=[
                    {'label': 'Show all wafers', 'value': 'all'},
                    {'label': 'Show only wafers with >10 RBS data points', 'value': 'filtered'}
                ],
                value='filtered',
                style={'marginBottom': '15px'}
            )
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("Display Options:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
            dcc.RadioItems(
                id='summary-radio',
                options=[
                    {'label': 'Show data summaries', 'value': 'show'},
                    {'label': 'Hide data summaries', 'value': 'hide'}
                ],
                value='hide',
                style={'marginBottom': '15px'}
            )
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),
        
        html.Div([
            html.Label("Scatter Plot Layer:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
            dcc.Dropdown(
                id='layer-dropdown',
                options=[{'label': layer, 'value': layer} for layer in unique_layers],
                value=unique_layers[0] if unique_layers else None,
                style={'marginBottom': '15px'}
            )
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),
        
        html.Div([
            html.Label("Scatter Plot CD Site:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
            dcc.Dropdown(
                id='cd-site-dropdown',
                options=[],
                value=None,
                style={'marginBottom': '15px'}
            )
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
    ], style={'textAlign': 'left', 'marginBottom': 20, 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
    
    # Display plots based on selections
    html.Div(id='plots-container', style={'textAlign': 'center'})
])

if __name__ == '__main__':
    app.run(debug=True, port=8052)