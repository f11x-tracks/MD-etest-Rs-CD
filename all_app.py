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

print(f"Loaded CD data: {len(df_cd)} rows")
print(f"FCCD data: {len(df_fccd)} rows")
print(f"DCCD data: {len(df_dccd)} rows")
print(f"Loaded RS data: {len(df_rs)} rows")
print(f"CD data columns: {df_cd.columns.tolist()}")
print(f"RS data columns: {df_rs.columns.tolist()}")

# Get common WAFER_IDs - matching WAFERID in CD data to WAFER_ID in RS data
cd_wafer_ids = set(df_cd['WAFERID'].unique())
rs_wafer_ids = set(df_rs['WAFER_ID'].unique())
common_wafer_ids = list(cd_wafer_ids & rs_wafer_ids)
print(f"Found {len(common_wafer_ids)} common WAFER_IDs")

# Filter data to only common wafers
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
        fig.update_layout(title=title, height=400)
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
        fig.update_layout(title=title, height=400)
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
        height=400,  # Back to original size for individual plots
        width=None,  # Let width be automatic
        margin=dict(l=40, r=40, t=50, b=40),  # Original margins
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Force equal aspect ratio
        font=dict(size=12)  # Original font size
    )
    
    return fig

# Create a function to generate plots based on filter criteria
def generate_plots(filter_option, show_summary):
    """Generate plots based on filter and display options"""
    plots = []
    plot_count = 0

    for (product, lot, layer), wafer_ids in wafer_groups.items():
        # Skip if layer is NaN
        if pd.isna(layer):
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
            
            valid_wafers.append((wafer_id, rs_data, fccd_data, dccd_data))
        
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
        
        # Create plots for each wafer in this group - each wafer on its own row
        for wafer_id, rs_data, fccd_data, dccd_data in valid_wafers:
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
                    dcc.Graph(figure=fig_rs, style={'height': '400px'})
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'})
            )
            
            # Add FCCD plot if available
            if fig_fccd is not None:
                plot_count_for_wafer += 1
                plot_divs.append(
                    html.Div([
                        dcc.Graph(figure=fig_fccd, style={'height': '400px'})
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
                )
            
            # Add DCCD plot if available
            if fig_dccd is not None:
                plot_count_for_wafer += 1
                margin_left = '2%' if fig_fccd is not None else '34%'  # Adjust margin based on FCCD presence
                plot_divs.append(
                    html.Div([
                        dcc.Graph(figure=fig_dccd, style={'height': '400px'})
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': margin_left})
                )
            
            # Create side-by-side layout for this wafer
            wafer_plots = html.Div(plot_divs, style={'marginBottom': '20px'})
            
            # Add data summary if enabled
            summary_section = html.Div()
            if show_summary == 'show':
                summary_section = html.Div([
                    html.H4(f"Data Summary for {wafer_id}", style={'textAlign': 'center'}),
                    html.Div([
                        html.Div([
                            html.P(f"{rbs_title} Data: {len(rs_data)} points"),
                            html.P(f"Range: {rs_data[rbs_column].min():.3f} - {rs_data[rbs_column].max():.3f}" if not rs_data.empty else "No data"),
                        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P(f"FCCD Data: {len(fccd_data)} points" if not fccd_data.empty else "FCCD: No data"),
                            html.P(f"Range: {fccd_data['CD'].min():.3f} - {fccd_data['CD'].max():.3f}" if not fccd_data.empty else "No data"),
                        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center', 'marginLeft': '5%'}),
                        
                        html.Div([
                            html.P(f"DCCD Data: {len(dccd_data)} points" if not dccd_data.empty else "DCCD: No data"),
                            html.P(f"Range: {dccd_data['CD'].min():.3f} - {dccd_data['CD'].max():.3f}" if not dccd_data.empty else "No data"),
                        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center', 'marginLeft': '5%'})
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

# Add callback for dynamic plot generation
@app.callback(
    Output('plots-container', 'children'),
    [Input('filter-radio', 'value'),
     Input('summary-radio', 'value')]
)
def update_plots(filter_option, show_summary):
    plots, plot_count = generate_plots(filter_option, show_summary)
    
    if not plots:
        return html.Div([
            html.H3("No wafers match the current filter criteria", 
                   style={'textAlign': 'center', 'marginTop': 50, 'color': 'gray'})
        ])
    
    # Add summary at the top
    summary_header = html.Div([
        html.H3(f"Displaying {plot_count} wafer plot pairs", 
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
                   style={'fontStyle': 'italic', 'fontSize': 14, 'margin': '5px 0'})
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
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
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
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
    ], style={'textAlign': 'left', 'marginBottom': 20, 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
    
    # Display plots based on selections
    html.Div(id='plots-container', style={'textAlign': 'center'})
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)