"""
@author: mareiss
Version: 1.0
"""

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dash_table
from dash.dash_table.Format import Format, Scheme

from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, kurtosis, skew, linregress

import astropy.units as u
from sunpy.coordinates.sun import carrington_rotation_time

from assets.fig_layout import my_figlayout, obs_linelayout, my_tablelayout
from utils.skill_scores import *

def create_line_chart(predicted_data, observed_data, baseline_data=None, property_type="Bulk Speed"):
    """
    Creates a time series comparison chart for predicted, observed, and baseline data.
    Observations are plotted as scatter points.
    """

    property_settings = {
        "Bulk Speed": {
            "yaxis_title": "Bulk Speed [km s⁻¹]",
            "title": "Solar Wind Speed Comparison",
            "y_range": [200, 900]
        },
        "Density": {
            "yaxis_title": "Number Density [cm⁻³]",
            "title": "Solar Wind Density Comparison"
        },
        "Temperature": {
            "yaxis_title": "Temperature [MK]",
            "title": "Solar Wind Temperature Comparison"
        },
        "vr": {
            "yaxis_title": "v<sub>r</sub> [km s⁻¹]",
            "title": "Solar Wind v<sub>r</sub> Comparison",
            "y_range": [200, 900]
        },
        "vt": {
            "yaxis_title": "v<sub>t</sub> [km s⁻¹]",
            "title": "Solar Wind v<sub>t</sub> Comparison"
        },
        "vp": {
            "yaxis_title": "v<sub>p</sub> [km s⁻¹]",
            "title": "Solar Wind v<sub>p</sub> Comparison"
        },
        "Br": {
            "yaxis_title": "B<sub>r</sub>",
            "title": "Polarity B<sub>r</sub> Comparison",
        },
        "Bt": {
            "yaxis_title": "B<sub>t</sub>",
            "title": "Polarity B<sub>t</sub> Comparison",
        },
        "Bp": {
            "yaxis_title": "B<sub>p</sub>",
            "title": "Polarity B<sub>p</sub> Comparison",
        }
    }

    settings = property_settings.get(property_type, property_settings["Bulk Speed"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=predicted_data["Time"],
        y=predicted_data["Model"],
        name=predicted_data["Name"].iloc[0],
        mode="lines",
        line=dict(width=2),
        hovertemplate="Time=%{x}<br>Model=%{y:.2f}<extra></extra>"
    ))

    if baseline_data is not None and not baseline_data.empty:
        fig.add_trace(go.Scatter(
            x=baseline_data["Time"],
            y=baseline_data["Model"],
            name=baseline_data["Name"].iloc[0],
            mode="lines",
            line=dict(color="lightgray", width=1.5, dash="dash"),
            hovertemplate="Time=%{x}<br>Baseline=%{y:.2f}<extra></extra>"
        ))

    fig.add_trace(go.Scatter(
        x=observed_data["Time"],
        y=observed_data["Obs"],
        name=observed_data["Name"].iloc[0],
        mode="markers",
        marker=dict(
            symbol="circle-open",
            size=2,
            color="#222222",
            line=dict(width=1)
        ),
        opacity=0.85,
        hovertemplate="Time=%{x}<br>Obs=%{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        my_figlayout,
        title=settings["title"],
        xaxis=dict(
            range=[
                predicted_data["Time"].iloc[0],
                predicted_data["Time"].iloc[-1]
            ]
        ),
        xaxis_title="Time (UTC)",
        yaxis_title=settings["yaxis_title"],
        legend=dict(orientation="h", x=0.35, y=1.15),
        height=400
    )

    if property_type in {"Br", "Bt", "Bp"}:
        fig.update_yaxes(
            range=[-1.2, 1.2],          
            tickmode="array",
            tickvals=[-1, 1],
            ticktext=["Negative", "Positive"],
            zeroline=False,
            zerolinecolor="lightgrey",
            showgrid=True
        )

    if settings.get("y_range"):
        fig.update_yaxes(range=settings["y_range"])

    return fig


def create_raincloud_plot(predicted_data, observed_data, property_type="Bulk Speed"):
    """
    Creates a horizontal raincloud-style plot combining violin and box plots 
    to compare the distribution of predicted and observed values.

            Parameters:
                    predicted_data (pandas.DataFrame or dict):
                        Model output to plot. Must contain:
                            - 'Time' (optional, datetime-like): timestamps
                            - 'Model' (float): predicted values
                            - 'Name' (str): model name (used in legend/labels)
                        If a dictionary is provided, it will be converted to a DataFrame.

                    observed_data (pandas.DataFrame or dict):
                        Observational reference data. Must contain:
                            - 'Time' (optional, datetime-like): timestamps
                            - 'Obs' (float): observed values
                            - 'Name' (str): dataset name (used in legend/labels)
                        If a dictionary is provided, it will be converted to a DataFrame.

                    property_type (str): 
                        The physical property being plotted.
                        Options: "Bulk Speed", "Density"

            Returns:
                    plotly.graph_objects.Figure:
                        Interactive Plotly figure showing:
                            - Half-violin plots for each dataset (predicted and observed)
                            - Box plots overlaid on the violins to highlight median, quartiles, and outliers
    """

    # Convert dictionaries to pandas DataFrames if necessary
    if not isinstance(predicted_data, pd.DataFrame):
        predicted_data = pd.DataFrame(predicted_data)
    if not isinstance(observed_data, pd.DataFrame):
        observed_data = pd.DataFrame(observed_data)
    
    # Correctly assign the columns for observed and predicted data
    predicted_data = predicted_data.rename(columns={'Model': 'Value'})
    predicted_data['Type'] = predicted_data['Name'][0]
    observed_data = observed_data.rename(columns={'Obs': 'Value'})
    observed_data['Type'] = observed_data['Name'][0]
    
    # Combine the data
    combined_data = pd.concat([predicted_data, observed_data], ignore_index=True)
    
    # Property-specific settings
    property_settings = {
        "Bulk Speed": {
            "title": "Distribution of Bulk Speed",
            "xaxis_title": "Bulk Speed [km s⁻¹]",
            "tick_spacing": 100
        },
        "Density": {
            "title": "Distribution of Number Density",
            "xaxis_title": "Number Density [cm⁻³]",
            "tick_spacing": 20
        },
        "Temperature": {
            "title": "Distribution of Temperature",
            "xaxis_title": "Temperature [MK]"
        },
        "vr": {
            "xaxis_title": "v<sub>r</sub> [km s⁻¹]",
            "title": "Solar Wind v<sub>r</sub> Comparison",
            "tick_spacing": 100
        },
        "vt": {
            "xaxis_title": "v<sub>t</sub> [km s⁻¹]",
            "title": "Solar Wind v<sub>t</sub> Comparison"
        },
        "vp": {
            "xaxis_title": "v<sub>p</sub> [km s⁻¹]",
            "title": "Solar Wind v<sub>p</sub> Comparison"
        }
    }

    settings = property_settings.get(property_type, property_settings["Bulk Speed"])

    # Raincloud plot
    fig = go.Figure()
    box_width = 0.15

    for data_type in combined_data['Type'].unique():
        subset = combined_data[combined_data['Type'] == data_type]

        # Half-violin plot
        fig.add_trace(go.Violin(
            x=subset['Value'],
            y=[data_type] * len(subset),
            name=data_type,
            points=False,
            showlegend=False,
            jitter=0.5,
            orientation='h',
            hoverinfo='skip',
            marker=dict(size=2),
            opacity=0.7
        ))

        # Box plot
        fig.add_trace(go.Box(
            x=subset['Value'],
            y=[data_type] * len(subset),
            name=data_type,
            boxmean=True,
            showlegend=False,
            marker_color='gray',
            orientation='h',
            boxpoints=False,
            width=box_width
        ))

    # Layout
    fig.update_layout(my_figlayout)

    fig.update_layout(
        title=settings["title"],
        xaxis_title=settings["xaxis_title"],
        height=500,
        xaxis=dict(
            showgrid=True,
            tick0=0,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            showgrid=False,
            tickangle=270, 
        )
    )

    return fig


def create_scatter_plot(predicted_data, observed_data, property_type="Bulk Speed", max_points=1000, opacity=0.4):
    """
    Creates a density-enhanced scatter plot comparing predicted vs. observed values.
    The plot overlays:
        - Kernel density estimation (KDE) contours showing point density
        - Downsampled scatter points for performance and visualization
        - A diagonal reference line
        - An optional linear regression fit with slope, intercept, and R² annotation

            Parameters:
                    predicted_data (pandas.DataFrame or dict):
                        Model output to plot. Must contain:
                            - 'Model' (float): predicted values
                            - 'Name' (str): model name (optional)
                        If a dictionary is provided, it will be converted to a DataFrame.

                    observed_data (pandas.DataFrame or dict):
                        Observational reference data. Must contain:
                            - 'Obs' (float): observed values
                            - 'Name' (str): dataset name (optional)
                        If a dictionary is provided, it will be converted to a DataFrame.

                    max_points (int, default=1000):
                        Maximum number of points to plot in the scatter layer.
                        Data is downsampled if it exceeds this number.

                    opacity (float, default=0.4):
                        Marker opacity for scatter points.

            Returns:
                    plotly.graph_objects.Figure:
                        Interactive Plotly figure containing:
                            - KDE contour density background
                            - Downsampled scatter points
                            - diagonal reference line
                            - Optional linear regression line
                            - Regression statistics annotation box
    """

    # Convert dictionaries to pandas DataFrames if necessary
    if not isinstance(predicted_data, pd.DataFrame):
        predicted_data = pd.DataFrame(predicted_data)
    if not isinstance(observed_data, pd.DataFrame):
        observed_data = pd.DataFrame(observed_data)

    # Combine observed and predicted into one DataFrame
    data = pd.DataFrame({
        'Observed': observed_data['Obs'],
        'Predicted': predicted_data['Model']
    }).dropna()

    # Downsample for scatter plot
    if len(data) > max_points:
        step = len(data) // max_points
        scatter_data = data.iloc[::step].head(max_points)
        downsampled = True
    else:
        scatter_data = data
        downsampled = False

    # Downsample for KDE
    if len(data) > max_points:
        step = len(data) // max_points
        kde_data = data.iloc[::step].head(max_points)
    else:
        kde_data = data

    # Property-specific settings
    property_settings = {
        "Bulk Speed": {
            "title": "Density and Scatter Plot: Bulk Speed",
            "unit": "km s⁻¹",
            "bounds_mode": "fixed",
            "min_val": 200,
            "max_val": 800,
            "tick0": 200,
            "dtick": 100,
        },
        "Density": {
            "title": "Density and Scatter Plot: Number Density",
            "unit": "cm⁻³",
            "bounds_mode": "p99",  
            "min_val": 0,
            "tick0": 0,
        },
        "Temperature": {
            "title": "Density and Scatter Plot: Temperature",
            "unit": "MK",
            "bounds_mode": "p99",  
            "min_val": 0,
            "fixed_dtick": 0.1,
        },
        "vr": {
            "title": "Density and Scatter Plot: v<sub>r</sub>",
            "unit": "km s⁻¹",
            "bounds_mode": "fixed",
            "min_val": 200,
            "max_val": 800,
            "tick0": 200,
            "dtick": 100,
        },
         "vt": {
            "title": "Density and Scatter Plot: v<sub>t</sub>",
            "unit": "km s⁻¹",
            "bounds_mode": "fixed",
            "min_val": -100,
            "max_val": 100,
            "tick0": 0,
            "dtick": 50,
        },
         "vp": {
            "title": "Density and Scatter Plot: v<sub>p</sub>",
            "unit": "km s⁻¹",
            "bounds_mode": "fixed",
            "min_val": -100,
            "max_val": 100,
            "tick0": 0,
            "dtick": 50,
        },
    }
    settings = property_settings.get(property_type, property_settings["Bulk Speed"])

    # Bounds
    if settings.get("bounds_mode") == "p99":
        # p99 of combined observed and predicted
        combined = pd.concat([data["Observed"], data["Predicted"]], ignore_index=True)
        combined = combined.astype(float).dropna()

        # guard against empty data
        if len(combined) == 0:
            min_val, max_val = 0, 1
        else:
            p99 = float(combined.quantile(0.99))
            max_val = max(p99, 1.0)
            min_val = 0

        raw = max_val / 10.0
        nice_steps = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        dtick = float(nice_steps[np.argmin(np.abs(nice_steps - raw))])
        tick0 = 0
    else:
        min_val, max_val = settings["min_val"], settings["max_val"]
        tick0 = settings["tick0"]
        dtick = settings["dtick"]


    if property_type == "Temperature":
        dtick = settings["fixed_dtick"]
        tick0 = settings.get("tick0", 0)

    fig = go.Figure()

    # Contour density background
    colorscale = [
        [0.0, 'rgba(255,255,255,0)'],
        [0.1, 'rgba(255,255,255,0)'],
        [0.3, 'rgba(255,200,100,0.2)'],
        [0.7, 'rgba(255,140,50,0.5)'],
        [1.0, 'rgba(255,100,0,0.8)']
    ]

    # Grid for KDE
    xi, yi = np.meshgrid(
        np.linspace(min_val, max_val, 100),
        np.linspace(min_val, max_val, 100)
    )

    # Evaluate KDE on grid
    kde = gaussian_kde(np.vstack([kde_data['Observed'], kde_data['Predicted']]))
    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

    fig.add_trace(go.Contour(
        x=xi[0],
        y=yi[:, 0],
        z=zi,
        showscale=False,
        colorscale=colorscale,
        hoverinfo='skip',
        contours=dict(
            showlines=True,
            start=zi.min(),
            end=zi.max(),
            size=(zi.max() - zi.min()) / 11
        ),
        line=dict(width=0.5, color='lightgrey')
    ))

    # Scatter points
    scatter_label = 'Data Points (downsampled)' if downsampled else 'Data Points'
    fig.add_trace(go.Scattergl(
        x=scatter_data['Observed'],
        y=scatter_data['Predicted'],
        mode='markers',
        marker=dict(size=3, color='black', opacity=opacity),
        name=scatter_label,
        showlegend=True
    ))

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='grey', dash='dash', width=1),
        name='Diagonal',
        showlegend=False
    ))

    # Linear regression line
    reg_stats = regressionStats(data['Predicted'], data['Observed'])
    slope = reg_stats['slope']
    intercept = reg_stats['intercept']
    r_squared = reg_stats['r_squared']

    reg_x = np.linspace(min_val, max_val, 100)
    reg_y = intercept + slope * reg_x

    fig.add_trace(go.Scatter(
        x=reg_x,
        y=reg_y,
        mode='lines',
        line=dict(color='black', dash='dashdot', width=1.5),
        name='Linear Regression',
        visible='legendonly'
    ))

    # Regression stats annotation
    fig.add_annotation(
        text=f"<b>Linear Regression:</b><br>"
             f"Slope: {slope:.2f}<br>"
             f"Intercept: {intercept:.2f}<br>"
             f"R²: {r_squared:.2f}",
        xref="paper", yref="paper",
        x=0.98, y=0.03,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=11, color="black"),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="lightgrey",
        borderwidth=1,
        opacity=0.8
    )

    unit = settings["unit"]
    obs_name = observed_data['Name'].iloc[0] if 'Name' in observed_data.columns else "Observation"
    pred_name = predicted_data['Name'].iloc[0] if 'Name' in predicted_data.columns else "Prediction"

    # Layout
    fig.update_layout(
        my_figlayout,
        title=settings["title"],
        xaxis_title=f"{obs_name} {property_type} [{unit}]",
        yaxis_title=f"{pred_name} {property_type} [{unit}]",
        xaxis=dict(
            range=[min_val, max_val],
            tick0=tick0,
            dtick=dtick,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            range=[min_val, max_val],
            tick0=tick0,
            dtick=dtick,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey'
        ),
        height=700,
        legend=dict(orientation='h', x=0.05, y=1.0)
    )

    return fig


def create_error_table_ptp(observed_data, predicted_data, baseline_data=None, property_type="Bulk Speed"):
    """
    Compute point-to-point error metrics comparing model predictions and (optionally)
    a baseline against observed data, and return the results as a Dash DataTable.

    Coverage rows:
      - Paired points [#]
      - Model points evaluated [%]
      - Observations covered [%]

    If baseline_data is None or empty, the Baseline column is not shown.
    """

    # Units depending on property type
    property_settings = {
        "Bulk Speed": {"unit": "km s⁻¹", "var_unit": "(km s⁻¹)²"},
        "Density": {"unit": "cm⁻³", "var_unit": "(cm⁻³)²"},
        "Temperature": {"unit": "MK", "var_unit": "(MK)²"},
        "vr": {"unit": "km s⁻¹", "var_unit": "(km s⁻¹)²"},
        "vp": {"unit": "km s⁻¹", "var_unit": "(km s⁻¹)²"},
        "vt": {"unit": "km s⁻¹", "var_unit": "(km s⁻¹)²"},
    }
    settings = property_settings.get(property_type, property_settings["Bulk Speed"])
    unit = settings["unit"]
    var_unit = settings["var_unit"]

    has_baseline = baseline_data is not None and not getattr(baseline_data, "empty", False)

    def _pair_obs_model(obs_df, model_df, obs_col="Obs", model_col="Model"):
        """Return paired DataFrame with columns ['Obs','Model'] plus counts and coverages."""
        n_obs_total = int(obs_df[obs_col].notna().sum()) if obs_col in obs_df else 0
        n_model_total = int(model_df[model_col].notna().sum()) if model_col in model_df else 0

        have_time = ("Time" in obs_df.columns) and ("Time" in model_df.columns)

        if have_time:
            obs_s = obs_df[["Time", obs_col]].dropna()
            mod_s = model_df[["Time", model_col]].dropna()
            paired = pd.merge(obs_s, mod_s, on="Time", how="inner").rename(
                columns={obs_col: "Obs", model_col: "Model"}
            )
        else:
            paired = pd.DataFrame(
                {"Obs": obs_df[obs_col], "Model": model_df[model_col]}
            ).dropna()

        n_paired = int(len(paired))
        cov_model_points_eval = (100.0 * n_paired / n_model_total) if n_model_total else 0.0
        cov_obs_covered = (100.0 * n_paired / n_obs_total) if n_obs_total else 0.0

        return paired, n_paired, n_obs_total, n_model_total, cov_model_points_eval, cov_obs_covered

    # Pair Model vs Obs
    paired_model, n_paired_m, n_obs_total, n_model_total, cov_m2o, cov_o2m = _pair_obs_model(
        observed_data, predicted_data, obs_col="Obs", model_col="Model"
    )

    # Pair Baseline vs Obs (if provided)
    if has_baseline:
        paired_base, n_paired_b, n_obs_total_b, n_base_total, cov_b2o, cov_o2b = _pair_obs_model(
            observed_data, baseline_data, obs_col="Obs", model_col="Model"
        )
        if n_obs_total_b and n_obs_total_b != n_obs_total:
            n_obs_total = n_obs_total_b

    # Compute metrics on paired data actually compared (Model)
    obs_vals = paired_model["Obs"]
    model_vals = paired_model["Model"]

    me = meScore(model_vals, obs_vals)
    mae = maeScore(model_vals, obs_vals)
    mse = meanSquaredError(model_vals, obs_vals)
    rms = rmsScore(model_vals, obs_vals)
    cc = ccScore(model_vals, obs_vals)
    msa = msaScore(model_vals, obs_vals)
    sspb = sspbScore(model_vals, obs_vals)
    mocr = mocrScore(model_vals, obs_vals)
    sdocr = sdocrScore(model_vals, obs_vals)

    # Baseline metrics (if provided)
    if has_baseline:
        obs_base = paired_base["Obs"]
        base_vals = paired_base["Model"]

        meBase = meScore(base_vals, obs_base)
        maeBase = maeScore(base_vals, obs_base)
        mseBase = meanSquaredError(base_vals, obs_base)
        rmsBase = rmsScore(base_vals, obs_base)
        ccBase = ccScore(base_vals, obs_base)
        msaBase = msaScore(base_vals, obs_base)
        sspbBase = sspbScore(base_vals, obs_base)
        mocrBase = mocrScore(base_vals, obs_base)
        sdocrBase = sdocrScore(base_vals, obs_base)

    table_data = [
        {"Metric": "Paired points [#]", "Model": n_paired_m},
        {"Metric": "Model points evaluated [%]", "Model": round(cov_m2o, 2)},
        {"Metric": "Observations covered [%]", "Model": round(cov_o2m, 2)},
    ]
    if has_baseline:
        table_data[0]["Baseline"] = n_paired_b
        table_data[1]["Baseline"] = round(cov_b2o, 2)
        table_data[2]["Baseline"] = round(cov_o2b, 2)

    metric_rows = [
        {'Metric': f'Mean Error [{unit}]', 'Model': round(me, 2)},
        {'Metric': f'Mean Absolute Error [{unit}]', 'Model': round(mae, 2)},
        {'Metric': f'Mean Squared Error [{var_unit}]', 'Model': round(mse, 2)},
        {'Metric': f'Root Mean Squared Error [{unit}]', 'Model': round(rms, 2)},
        {'Metric': 'Pearson Correlation Coefficient', 'Model': round(cc, 2)},
        {'Metric': 'Median Symmetric Accuracy [%]', 'Model': round(msa, 2)},
        {'Metric': 'Symmetric Signed Percentage Bias [%]', 'Model': round(sspb, 2)},
        {'Metric': 'Mean O/C Ratio', 'Model': round(mocr, 2)},
        {'Metric': 'SD of O/C Ratio', 'Model': round(sdocr, 2)}
    ]

    if has_baseline:
        baseline_vals = [
            round(meBase, 2),
            round(maeBase, 2),
            round(mseBase, 2),
            round(rmsBase, 2),
            round(ccBase, 2),
            round(msaBase, 2),
            round(sspbBase, 2),
            round(mocrBase, 2),
            round(sdocrBase, 2),
        ]
        for row, b in zip(metric_rows, baseline_vals):
            row['Baseline'] = b

    table_data.extend(metric_rows)

    columns = [
        {'name': 'Metric', 'id': 'Metric'},
        {'name': predicted_data['Name'][0], 'id': 'Model', 'type': 'numeric',
         'format': Format(precision=2, scheme=Scheme.fixed)}
    ]
    if has_baseline:
        columns.append(
            {'name': 'Baseline Model', 'id': 'Baseline', 'type': 'numeric',
             'format': Format(precision=2, scheme=Scheme.fixed)}
        )

    style_cell_conditional = [
        {'if': {'column_id': 'Model'}, 'textAlign': 'right'},
    ]
    if has_baseline:
        style_cell_conditional.append({'if': {'column_id': 'Baseline'}, 'textAlign': 'right'})

    # Create the data table
    error_table = dash_table.DataTable(
        data=table_data,
        columns=columns,
        style_cell_conditional=style_cell_conditional,
        **my_tablelayout
    )

    return error_table


def create_residual_plot(predicted_data, observed_data, property_type="Bulk Speed", max_points=1000, opacity=0.4):
    """
    Creates a residual plot showing the difference between observed and predicted values.
    The plot includes:
        - Kernel density estimation (KDE) contours for residual distribution
        - Downsampled scatter points for large datasets
        - A horizontal zero-reference line for bias detection

            Parameters:
                    predicted_data (pandas.DataFrame or dict):
                        Predicted model data. Must contain:
                            - 'Model' (float): predicted values
                            - 'Name' (str): model name (optional)
                        If a dictionary is provided, it will be converted to a DataFrame.

                    observed_data (pandas.DataFrame or dict):
                        Observed reference data. Must contain:
                            - 'Obs' (float): observed values
                            - 'Name' (str): dataset name (optional)
                        If a dictionary is provided, it will be converted to a DataFrame.

                    max_points (int, default=1000):
                        Maximum number of points plotted in the scatter layer.
                        Data is evenly downsampled if it exceeds this number 
                        (improves performance for large datasets).

                    opacity (float, default=0.4):
                        Marker opacity for scatter points.

            Returns:
                    plotly.graph_objects.Figure:
                        Interactive Plotly figure containing:
                            - KDE contour background showing residual density
                            - Downsampled scatter points of residuals vs. observed values
                            - Horizontal zero-reference line
                            - Configurable axis ranges

            Notes:
                    - Positive residuals mean the model underestimates values; negative residuals indicate overestimation.
    """

    # Convert dictionaries to pandas DataFrames if necessary
    if not isinstance(predicted_data, pd.DataFrame):
        predicted_data = pd.DataFrame(predicted_data)
    if not isinstance(observed_data, pd.DataFrame):
        observed_data = pd.DataFrame(observed_data)

    # Combine and compute residuals
    data = pd.DataFrame({
        'Observed': observed_data['Obs'],
        'Predicted': predicted_data['Model']
    }).dropna()
    data['Residuals'] = data['Observed'] - data['Predicted']

    # Downsample for scatter and KDE
    if len(data) > max_points:
        step = len(data) // max_points
        scatter_data = data.iloc[::step].head(max_points)
        kde_data = scatter_data
        downsampled = True
    else:
        scatter_data = kde_data = data
        downsampled = False

    # Property-specific settings
    property_settings = {
        "Bulk Speed": {
            "title": "Residuals vs. Observed Values: Bulk Speed",
            "unit": "km s⁻¹",
            "x_mode": "fixed",
            "x_min": 200,
            "x_max": 800,
            "x_tick0": 200,
            "x_dtick": 100,
            "y_mode": "fixed",
            "y_min": -500,
            "y_max": 500,
            "y_tick0": -500,
            "y_dtick": 200,
        },
        "Density": {
            "title": "Residuals vs. Observed Values: Number Density",
            "unit": "cm⁻³",
            "x_mode": "p99",
            "y_mode": "p99_abs",   
            "x_min": 0,
            "x_tick0": 0,
        },
        "Temperature": {
            "title": "Residuals vs. Observed Values: Temperature",
            "unit": "MK",
            "x_mode": "p99",
            "y_mode": "p99_abs",   
            "x_min": 0,
            "x_tick0": 0,
            "fixed_dtick": 0.1,
        },
        "vr": {
            "title": "Residuals vs. Observed Values: v<sub>r</sub>",
            "unit": "km s⁻¹",
            "x_mode": "fixed",
            "x_min": 200,
            "x_max": 800,
            "x_tick0": 200,
            "x_dtick": 100,
            "y_mode": "fixed",
            "y_min": -500,
            "y_max": 500,
            "y_tick0": -500,
            "y_dtick": 200,
        },
        "vt": {
            "title": "Residuals vs. Observed Values: v<sub>t</sub>",
            "unit": "km s⁻¹",
            "x_mode": "fixed",
            "x_min": -100,
            "x_max": 100,
            "x_tick0": 0,
            "x_dtick": 50,
            "y_mode": "fixed",
            "y_min": -100,
            "y_max": 100,
            "y_tick0": 0,
            "y_dtick": 50,
        },
        "vp": {
            "title": "Residuals vs. Observed Values: v<sub>p</sub>",
            "unit": "km s⁻¹",
            "x_mode": "fixed",
            "x_min": -100,
            "x_max": 100,
            "x_tick0": 0,
            "x_dtick": 50,
            "y_mode": "fixed",
            "y_min": -100,
            "y_max": 100,
            "y_tick0": 0,
            "y_dtick": 50,
        },
    }
    settings = property_settings.get(property_type, property_settings["Bulk Speed"])
    unit = settings["unit"]

    # Bounds
    if settings.get("x_mode") == "p99":
        combined_x = pd.concat([data["Observed"], data["Predicted"]], ignore_index=True).astype(float).dropna()
        p99_x = float(combined_x.quantile(0.99)) if len(combined_x) else 1.0
        x_min = settings.get("x_min", 0)
        x_max = max(p99_x, x_min + 1.0)

        raw = x_max / 10.0
        nice_steps = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        x_dtick = float(nice_steps[np.argmin(np.abs(nice_steps - raw))])
        x_tick0 = settings.get("x_tick0", 0)
    else:
        x_min, x_max = settings["x_min"], settings["x_max"]
        x_tick0, x_dtick = settings["x_tick0"], settings["x_dtick"]

    if settings.get("y_mode") == "p99_abs":
        abs_res = data["Residuals"].astype(float).abs().dropna()
        p99_r = float(abs_res.quantile(0.99)) if len(abs_res) else 1.0

        y_max = max(p99_r, 1.0)
        y_min = -y_max

        NICE_STEPS = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])

        target_step = y_max / 6.0   # gives ~8 ticks total (-4 … +4)
        y_dtick = float(NICE_STEPS[NICE_STEPS >= target_step][0])

        y_tick0 = 0
    else:
        y_min, y_max = settings["y_min"], settings["y_max"]
        y_tick0, y_dtick = settings["y_tick0"], settings["y_dtick"]

    if property_type == "Temperature":
        x_dtick = settings["fixed_dtick"]
        x_tick0 = settings.get("x_tick0", 0)
        y_dtick = settings["fixed_dtick"]
        y_tick0 = 0  #symmetry around zero

    # Grid and KDE
    xi, yi = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    values = np.vstack([kde_data['Observed'], kde_data['Residuals']])
    kde = gaussian_kde(values)
    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

    colorscale = [
        [0.0, 'rgba(255,255,255,0)'],
        [0.1, 'rgba(255,255,255,0)'],
        [0.3, 'rgba(255,200,100,0.2)'],
        [0.7, 'rgba(255,140,50,0.5)'],
        [1.0, 'rgba(255,100,0,0.8)']
    ]

    fig = go.Figure()

    # Contour background
    fig.add_trace(go.Contour(
        x=xi[0],
        y=yi[:, 0],
        z=zi,
        showscale=False,
        colorscale=colorscale,
        hoverinfo='skip',
        contours=dict(
            showlines=True,
            start=zi.min(),
            end=zi.max(),
            size=(zi.max() - zi.min()) / 11
        ),
        line=dict(width=0.5, color='lightgrey')
    ))

    # Scatter points
    fig.add_trace(go.Scattergl(
        x=scatter_data['Observed'],
        y=scatter_data['Residuals'],
        mode='markers',
        marker=dict(size=3, color='black', opacity=opacity),
        name='Data Points (downsampled)' if downsampled else 'Data Points',
        showlegend=True
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="grey")

    obs_name = observed_data['Name'].iloc[0] if 'Name' in observed_data.columns else "Observation"

    fig.update_layout(
        my_figlayout,
        title=settings["title"],
        xaxis_title=f"{obs_name} {property_type} [{unit}]",
        yaxis_title=f"Residuals [{unit}]",
        xaxis=dict(
            range=[x_min, x_max],
            tick0=x_tick0,
            dtick=x_dtick,
            showgrid=False,
            gridwidth=1,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            range=[y_min, y_max],
            tick0=y_tick0,
            dtick=y_dtick,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        ),
        height=700,
        legend=dict(orientation='h', x=0.05, y=1.0)
    )

    return fig


def create_statistics_table(observed_data,predicted_data, baseline_data=None, property_type="Bulk Speed"):
    """
    Creates a formatted statistics table comparing observed, predicted, and baseline
    distributions using a Dash DataTable.

    Parameters
    ----------
    observed_data, predicted_data, baseline_data : pandas.DataFrame
        Must include columns:
          - observed_data: 'Obs', 'Name'
          - predicted_data: 'Model', 'Name'
          - baseline_data (optional): 'Model'
    property_type : str
        "Bulk Speed" or "Density"

    If baseline_data is None or empty, the Baseline column is not shown.
    """

    observed_values = observed_data['Obs']
    predicted_values = predicted_data['Model']

    has_baseline = baseline_data is not None and not getattr(baseline_data, "empty", False)

    obs_stats = descriptiveStats(observed_values)
    pred_stats = descriptiveStats(predicted_values)

    if has_baseline:
        baseline_values = baseline_data['Model']
        base_stats = descriptiveStats(baseline_values)

    # Property-specific settings
    property_settings = {
        "Bulk Speed": {
            "unit": "km s⁻¹",
            "variance_unit": "(km s⁻¹)²",
        },
        "Density": {
            "unit": "cm⁻³",
            "variance_unit": "(cm⁻³)²",
        },
        "Temperature": {
            "unit": "MK",
            "variance_unit": "(MK)²",
        },
        "vr": {
            "unit": "km s⁻¹",
            "variance_unit": "(km s⁻¹)²",
        },
        "vt": {
            "unit": "km s⁻¹",
            "variance_unit": "(km s⁻¹)²",
        },
        "vp": {
            "unit": "km s⁻¹",
            "variance_unit": "(km s⁻¹)²",
        },
    }
    settings = property_settings.get(property_type, property_settings["Bulk Speed"])
    unit = settings["unit"]
    var_unit = settings["variance_unit"]

    # Labels
    label_map = {
        'mean': f'Mean [{unit}]',
        'median': f'Median [{unit}]',
        'std': f'SD [{unit}]',
        'min': f'Minimum [{unit}]',
        'max': f'Maximum [{unit}]',
        'var': f'Variance [{var_unit}]',
        'skewness': 'Skewness',
        'kurtosis': 'Kurtosis'
    }

    data = []
    for key, label in label_map.items():
        row = {
            '': label,
            'Observation': obs_stats.get(key, 'NaN'),
            'Prediction': pred_stats.get(key, 'NaN'),
        }
        if has_baseline:
            row['Baseline'] = base_stats.get(key, 'NaN')
        data.append(row)

    # Build columns
    columns = [
        {'name': '', 'id': ''},
        {'name': observed_data['Name'][0], 'id': 'Observation', 'type': 'numeric',
         'format': Format(precision=2, scheme=Scheme.fixed)},
        {'name': predicted_data['Name'][0], 'id': 'Prediction', 'type': 'numeric',
         'format': Format(precision=2, scheme=Scheme.fixed)},
    ]
    if has_baseline:
        columns.append(
            {'name': 'Baseline', 'id': 'Baseline', 'type': 'numeric',
             'format': Format(precision=2, scheme=Scheme.fixed)}
        )

    style_cell_conditional = [
        {'if': {'column_id': 'Observation'}, 'textAlign': 'right'},
        {'if': {'column_id': 'Prediction'}, 'textAlign': 'right'},
    ]
    if has_baseline:
        style_cell_conditional.append({'if': {'column_id': 'Baseline'}, 'textAlign': 'right'})

    statistics_table = dash_table.DataTable(
        data=data,
        columns=columns,
        style_cell_conditional=style_cell_conditional,
        **my_tablelayout
    )

    return statistics_table


def create_rmse_timeline(predicted_data, observed_data, benchmark_data=None,
                         property_type="Bulk Speed",
                         normalize_time="none", floor="H"):
    """
    Creates an interactive timeline plot showing RMSE of a model and (optionally) a benchmark
    over consecutive Carrington rotations.
    """

    # Property-specific settings
    property_settings = {
        "Bulk Speed": {
            "unit": "km s⁻¹",
            "abs_dtick": 50,
            "abs_tick0": 0,
        },
        "Density": {
            "unit": "cm⁻³",
            "abs_dtick": 20,
            "abs_tick0": 0,
        },
        "Temperature": {
            "unit": "MK",
            "abs_dtick": 0.1,
            "abs_tick0": 0,
        },
        "vr": {
            "unit": "km s⁻¹",
            "abs_dtick": 50,
            "abs_tick0": 0,
        },
        "vt": {
            "unit": "km s⁻¹",
            "abs_dtick": 5,
            "abs_tick0": 0,
        },
        "vp": {
            "unit": "km s⁻¹",
            "abs_dtick": 5,
            "abs_tick0": 0,
        },
    }
    settings = property_settings.get(property_type, property_settings["Bulk Speed"])
    unit = settings["unit"]

    def _normalize_time(df, col="Time"):
        if normalize_time == "none":
            d = df.copy()
            d[col] = pd.to_datetime(d[col], errors="coerce")
            return d.dropna(subset=[col]).sort_values(col)

        t = pd.to_datetime(df[col], utc=True, errors="coerce")
        if floor:
            t = t.dt.floor(floor)

        if normalize_time == "naive_utc":
            t = t.dt.tz_convert("UTC").dt.tz_localize(None)
        elif normalize_time == "aware_utc":
            t = t.dt.tz_convert("UTC")
        else:
            raise ValueError('normalize_time must be "none", "naive_utc", or "aware_utc"')

        d = df.copy()
        d[col] = t
        return d.dropna(subset=[col]).sort_values(col)

    # Normalize times
    predicted_data = _normalize_time(predicted_data, "Time")
    observed_data  = _normalize_time(observed_data,  "Time")

    has_benchmark = benchmark_data is not None and not getattr(benchmark_data, "empty", False)
    if has_benchmark:
        benchmark_data = _normalize_time(benchmark_data, "Time")

    # Ensure time is datetime
    observed_data['Time'] = pd.to_datetime(observed_data['Time'])

    # Merge predictions with observations
    merged_main = pd.merge(predicted_data, observed_data, on='Time', how='inner')
    if has_benchmark:
        merged_bench = pd.merge(benchmark_data, observed_data, on='Time', how='inner')

    # Carrington rotation range
    start_time = observed_data['Time'].min()
    end_time   = observed_data['Time'].max()
    cr_start = 1 + int((start_time - pd.Timestamp("1853-11-09")) / pd.Timedelta(days=27.2753))
    cr_end   = int((end_time   - pd.Timestamp("1853-11-09")) / pd.Timedelta(days=27.2753))

    model_name = predicted_data['Name'].iloc[0]
    benchmark_name = benchmark_data['Name'].iloc[0] if has_benchmark else None

    metrics = ['RMSE']
    error_main = {m: [] for m in metrics}
    error_bench = {m: [] for m in metrics} if has_benchmark else None
    error_rel = {m: [] for m in metrics} if has_benchmark else None
    cr_numbers = []

    def _rmse(a, b):
        a = pd.to_numeric(pd.Series(a), errors="coerce")
        b = pd.to_numeric(pd.Series(b), errors="coerce")
        m = (a - b) ** 2
        return float(np.sqrt(np.nanmean(m)))

    for cr in range(cr_start, cr_end + 1):
        cr_start_time = carrington_rotation_time(cr, 360 * u.deg).datetime
        cr_end_time   = carrington_rotation_time(cr,   1 * u.deg).datetime

        main = merged_main[(merged_main['Time'] >= cr_start_time) & (merged_main['Time'] < cr_end_time)]
        if has_benchmark:
            bench = merged_bench[(merged_bench['Time'] >= cr_start_time) & (merged_bench['Time'] < cr_end_time)]

            if not main.empty and not bench.empty:
                model, obs, base = main['Model'], main['Obs'], bench['Model']

                rmse_m = _rmse(model, obs)
                rmse_b = _rmse(base,  obs)

                error_main['RMSE'].append(rmse_m)
                error_bench['RMSE'].append(rmse_b)
                error_rel['RMSE'].append(100 * (rmse_b - rmse_m) / rmse_b if rmse_b else np.nan)

                cr_numbers.append(cr)
        else:
            if not main.empty:
                model, obs = main['Model'], main['Obs']
                rmse_m = _rmse(model, obs)
                error_main['RMSE'].append(rmse_m)
                cr_numbers.append(cr)

    # Build plot
    fig = go.Figure()

    if has_benchmark:
        for metric in metrics:
            fig.add_trace(go.Scatter(
                x=cr_numbers, y=error_main[metric],
                name=f"{metric} - {model_name}", mode='lines+markers'
            ))
            fig.add_trace(go.Scatter(
                x=cr_numbers, y=error_bench[metric],
                name=f"{metric} - {benchmark_name}", mode='lines+markers',
                line=dict(dash='dash', color='grey')
            ))
            fig.add_trace(go.Scatter(
                x=cr_numbers, y=error_rel[metric],
                name=f"{metric} - Relative [%]", mode='lines+markers'
            ))
            # zero line for relative view
            fig.add_trace(go.Scatter(
                x=cr_numbers, y=[0] * len(cr_numbers),
                name=f"{benchmark_name}", mode='lines',
                line=dict(dash='dash', color='grey'),
                showlegend=False
            ))

        fig.data[0].visible = True
        fig.data[1].visible = True
        fig.data[2].visible = False
        fig.data[3].visible = False

        abs_button = dict(
            label="Absolute",
            method="update",
            args=[
                {"visible": [True, True, False, False]},
                {"yaxis": {
                    "title": f"RMSE [{unit}]",
                    "tick0": settings["abs_tick0"],
                    "dtick": settings["abs_dtick"],
                    "showgrid": True,
                    "gridwidth": 0.5,
                    "gridcolor": "lightgrey"
                }, "annotations": []}
            ]
        )
        rel_button = dict(
            label="Relative [%]",
            method="update",
            args=[
                {"visible": [False, False, True, True]},
                {"yaxis": {
                    "title": "Relative Skill [%]",
                    "tick0": 0,
                    "dtick": 10,
                    "showgrid": True,
                    "gridwidth": 0.5,
                    "gridcolor": "lightgrey"
                },
                 "annotations": [dict(
                     text="Above zero: better than benchmark<br>Below zero: worse than benchmark",
                     xref="paper", yref="paper",
                     x=0.98, y=0.15,
                     xanchor="right", yanchor="top",
                     showarrow=False,
                     font=dict(size=11, color="black"),
                     bgcolor="rgba(255, 255, 255, 0.8)",
                     bordercolor="lightgrey",
                     borderwidth=1,
                     opacity=0.9
                 )]}
            ]
        )

        updatemenus = [
            dict(
                buttons=[abs_button, rel_button],
                direction="down",
                showactive=True,
                x=0.01, xanchor="left",
                y=1.05, yanchor="top"
            )
        ]
    else:
        fig.add_trace(go.Scatter(
            x=cr_numbers, y=error_main['RMSE'],
            name=f"RMSE - {model_name}", mode='lines+markers'
        ))
        updatemenus = []

    fig.update_layout(
        my_figlayout,
        title=f"Skill Over Time at {observed_data['Name'].iloc[0]}",
        xaxis_title="Carrington Rotation Number",
        yaxis_title=f"RMSE [{unit}]",
        height=500,
        xaxis=dict(
            tickmode='linear',
            tick0=cr_numbers[0] if cr_numbers else 0,
            dtick=1,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            tick0=settings["abs_tick0"],
            dtick=settings["abs_dtick"],
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey'
        ),
        showlegend=True,
        legend=dict(orientation='h', x=0.34, y=1.12),
        updatemenus=updatemenus
    )

    return fig

def create_binary_classification_plot(predicted_data, observed_data, threshold=400, property_type="Bulk Speed"):
    """
    Creates a two-row combined plot to compare time-series (predicted vs. observed)
    and visualize binary event classifications over the same time window.

    Row 1:
        - Lines for predicted model values and observed values
        - A horizontal threshold line at the specified speed

    Row 2:
        - A categorical timeline of event outcomes (Hit, Miss, False Alarm, Correct Rejection)
          placed at their corresponding timestamps

            Parameters:
                    predicted_data (pandas.DataFrame):
                        Model predictions. Must contain:
                            - 'Time'   (datetime-like): timestamps (consistent tz handling recommended)
                            - 'Model'  (float): predicted bulk speeds [km s⁻¹]
                            - 'Name'   (str): model name (used in legend)
                    observed_data (pandas.DataFrame):
                        Observations. Must contain:
                            - 'Time'   (datetime-like): timestamps aligned to model or later interpolated
                            - 'Obs'    (float): observed bulk speeds [km s⁻¹]
                            - 'Name'   (str, optional): not required here
                    threshold (float, default=400):
                        Threshold (in [km s⁻¹]) used by `classifyBinaryEvents()` to define an “event”.

            Returns:
                    plotly.graph_objects.Figure:
                        A two-row subplot with shared x-axis:
                          - Row 1: Time-series lines for Model and Observed + threshold line
                          - Row 2: Colored markers per timestamp showing the event outcome
    """

    # Property-specific
    property_settings = {
        "Bulk Speed": {
            "yaxis_title": "Bulk Speed [km s⁻¹]",
            "title": "Binary Classification Comparison – Solar Wind Speed",
            "show_threshold": True,
            "y_range": [200, 900],
        },
        "Density": {
            "yaxis_title": "Number Density [cm⁻³]",
            "title": "Binary Classification Comparison – Number Density",
            "show_threshold": True,
        },
        "Temperature": {
            "yaxis_title": "Temperature [MK]",
            "title": "Binary Classification Comparison – Temperature",
            "show_threshold": True,
        },
        "vr": {
            "yaxis_title": "v<sub>r</sub> [km s⁻¹]",
            "title": "Binary Classification Comparison – v<sub>r</sub>",
            "show_threshold": True,
            "y_range": [200, 900],
        },
        "vt": {
            "yaxis_title": "v<sub>t</sub> [km s⁻¹]",
            "title": "Binary Classification Comparison – v<sub>t</sub>",
            "show_threshold": True,
        },
        "vp": {
            "yaxis_title": "v<sub>p</sub> [km s⁻¹]",
            "title": "Binary Classification Comparison – v<sub>p</sub>",
            "show_threshold": True,
        },
        "Br": {
            "yaxis_title": "B<sub>r</sub>",
            "title": "Binary Classification Comparison – Polarity B<sub>r</sub>",
            "show_threshold": False,
            "polarity_axis": True,
        },
        "Bt": {
            "yaxis_title": "B<sub>t</sub>",
            "title": "Binary Classification Comparison – Polarity B<sub>t</sub>",
            "show_threshold": False,
            "polarity_axis": True,
        },
        "Bp": {
            "yaxis_title": "B<sub>p</sub>",
            "title": "Binary Classification Comparison – Polarity B<sub>p</sub>",
            "show_threshold": False,
            "polarity_axis": True,
        }
    }

    settings = property_settings.get(property_type, property_settings["Bulk Speed"])

    outcomes, _ = classifyBinaryEvents(predicted_data, observed_data, threshold)

    outcome_df = pd.DataFrame({
        "Time": predicted_data["Time"],
        "Model": predicted_data["Model"],
        "Obs": observed_data["Obs"],
        "Outcome": outcomes
    })

    outcome_colors = {
        "Hit": "green",
        "Miss": "red",
        "False Alarm": "orange",
        "Correct Rejection": "blue",
    }

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.8, 0.2]
    )

    # Time series
    fig.add_trace(go.Scatter(
        x=predicted_data["Time"],
        y=predicted_data["Model"],
        mode="lines",
        name=predicted_data["Name"].iloc[0],
        line=dict(width=2),
        hovertemplate="Time=%{x}<br>Model=%{y:.2f}<extra></extra>"
    ), row=1, col=1)

    # Observations
    fig.add_trace(go.Scatter(
        x=observed_data["Time"],
        y=observed_data["Obs"],
        mode="markers",
        name=observed_data["Name"].iloc[0],
        marker=dict(
            symbol="circle-open",
            size=2,
            color="#222222",
            line=dict(width=1)
        ),
        opacity=0.85,
        hovertemplate="Time=%{x}<br>Obs=%{y:.2f}<extra></extra>"
    ), row=1, col=1)

    # Threshold
    if settings.get("show_threshold", True):
        fig.add_shape(
            type="line",
            x0=predicted_data["Time"].iloc[0],
            x1=predicted_data["Time"].iloc[-1],
            y0=threshold,
            y1=threshold,
            line=dict(color="black", width=1),
            row=1,
            col=1
        )

    # Event outcomes
    for outcome in outcome_df["Outcome"].unique():
        df_subset = outcome_df[outcome_df["Outcome"] == outcome]

        fig.add_trace(go.Scatter(
            x=df_subset["Time"],
            y=[outcome] * len(df_subset),
            mode="markers",
            name=outcome,
            marker=dict(
                color=outcome_colors.get(outcome, "grey"),
                symbol="circle",
                size=5
            )
        ), row=2, col=1)

    fig.update_layout(my_figlayout)

    fig.update_layout(
        title=settings["title"],
        yaxis_title=settings["yaxis_title"],
        xaxis=dict(
            range=[
                predicted_data["Time"].iloc[0],
                predicted_data["Time"].iloc[-1]
            ]
        ),
        legend=dict(orientation='h', x=0.1, y=1.02),  
        height=500
    )

    # Optional y-range
    if settings.get("y_range"):
        fig.update_yaxes(range=settings["y_range"], row=1, col=1)

    # Polarity axis styling
    if property_type in {"Br", "Bt", "Bp"}:
        fig.update_yaxes(
            range=[-0.2, 1.2],
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["Negative", "Positive"],
            zeroline=False,
            showgrid=True,
            row=1,
            col=1
        )

    # Outcome y-axis
    fig.update_yaxes(
        tickvals=outcome_df["Outcome"].unique(),
        ticktext=outcome_df["Outcome"].unique(),
        row=2,
        col=1
    )
    return fig


def create_cumulative_binary_outcomes(predicted_data, observed_data, threshold=400):
    """
    Creates a cumulative stacked area chart showing binary event outcomes 
    (Hit, Miss, False Alarm, Correct Rejection) over time.

            Parameters:
                    predicted_data (pandas.DataFrame):
                        Model predictions. Must contain:
                            - 'Time' (datetime-like): timestamps
                            - Model variable column required by `classifyBinaryEvents`
                    
                    observed_data (pandas.DataFrame):
                        Observational reference data. Must contain:
                            - 'Time' (datetime-like): timestamps
                            - Observation variable column required by `classifyBinaryEvents`
                    
                    threshold (float, optional):
                        Threshold value for binary event classification.
            
            Returns:
                    plotly.graph_objects.Figure:
                        Interactive stacked area chart with cumulative counts for:
                            - Hit (true positive)
                            - Miss (false negative)
                            - False Alarm (false positive)
                            - Correct Rejection (true negative)
            
            Notes:
                    - 'Correct Rejection' is hidden by default in the legend.
    """

    outcome_colors = {
        'Hit': 'green',
        'Miss': 'red',
        'False Alarm': 'orange',
        'Correct Rejection': 'blue'
    }

    outcomes, _ = classifyBinaryEvents(predicted_data, observed_data, threshold)
    outcome_df = pd.DataFrame({'Time': predicted_data['Time'], 'Outcome': outcomes})
    
    # Ensure all outcomes are represented at each time step, even if they are zero
    all_outcomes = ['Hit', 'Miss', 'False Alarm', 'Correct Rejection']
    outcome_df['Count'] = 1
    outcome_counts = outcome_df.pivot_table(index='Time', columns='Outcome', values='Count', aggfunc='sum', fill_value=0)
    
    for outcome in all_outcomes:
        if outcome not in outcome_counts.columns:
            outcome_counts[outcome] = 0
    
    # Calculate cumulative sums
    outcome_counts_cumulative = outcome_counts.cumsum()
    outcome_counts_cumulative.reset_index(inplace=True)

    fig = go.Figure()

    # Add a stacked area trace for each outcome category
    for outcome in all_outcomes:
        fig.add_trace(go.Scatter(
            x=outcome_counts_cumulative['Time'],
            y=outcome_counts_cumulative[outcome],
            mode='lines',
            name=outcome,
            stackgroup='one',
            line_color=outcome_colors[outcome],
            visible='legendonly' if outcome == 'Correct Rejection' else True
        ))

    # Update the figure with my_figlayout
    fig.update_layout(my_figlayout)

    # Apply custom layout if defined
    fig.update_layout(
        title='Cumulative Outcomes Over Time', 
        xaxis_title='Time', 
        yaxis_title='Cumulative Count', 
        legend=dict(x=0.05, y=1.0),
        height=500
    )

    return fig


def create_error_table_binary(observed_data, predicted_data, baseline_data, threshold):
    """
    Creates a styled Dash DataTable comparing binary classification 
    metrics for a model and a baseline, based on thresholded 
    continuous data.

    Parameters:
        observed_data (pandas.DataFrame):
            Reference observation dataset. Must contain:
                - 'Time' (datetime-like): timestamps
                - 'Obs': observed values
                - 'Name': dataset name (string)
        
        predicted_data (pandas.DataFrame):
            Model prediction dataset. Must contain:
                - 'Time' (datetime-like): timestamps
                - 'Model': predicted values
                - 'Name': model name (string)
        
        baseline_data (pandas.DataFrame):
            Baseline or benchmark prediction dataset. Must contain:
                - 'Time' (datetime-like): timestamps
                - 'Model': predicted values
                - 'Name': baseline name (string)
        
        threshold (float):
            Threshold applied to observed and predicted data 
            to classify binary events.

    Returns:
        dash_table.DataTable:
            A Dash DataTable displaying binary classification 
            metrics for both the model and the baseline. 
            Columns include:
                - True Positive Rate (TPR)
                - False Positive Rate (FPR)
                - Threat Score (TS)
                - True Skill Statistics (TSS)
                - Bias

    Notes:
        - All datasets must be time-aligned and share the 
          same timezone before calling this function.
        - The classification metrics are computed from 
          contingency tables produced by `classifyBinaryEvents`.
        - The DataTable uses number formatting with fixed 
          precision (2 decimal places).
    """

    # Extract names
    model_name = predicted_data['Name'][0]
    baseline_name = baseline_data['Name'][0]

    # Contingency tables
    baseline_as_obs = baseline_data[['Model']].rename(columns={'Model': 'Obs'})
    _, ct = classifyBinaryEvents(predicted_data, observed_data, threshold)
    _, ct_base = classifyBinaryEvents(predicted_data, baseline_as_obs, threshold)

    # Model metrics
    tpr = truePositiveRate(ct)
    fpr = falsePostiveRate(ct)
    ts = threatScore(ct)
    tss = trueSkillStatistics(ct)
    bi = bias(ct)

    # Baseline metrics
    tpr_base = truePositiveRate(ct_base)
    fpr_base = falsePostiveRate(ct_base)
    ts_base = threatScore(ct_base)
    tss_base = trueSkillStatistics(ct_base)
    bi_base = bias(ct_base)

    # Table data
    data = [
        {'Metric': 'True Positive Rate', model_name: tpr[0], baseline_name: tpr_base[0]},
        {'Metric': 'False Positive Rate', model_name: fpr[0], baseline_name: fpr_base[0]},
        {'Metric': 'Threat Score', model_name: ts[0], baseline_name: ts_base[0]},
        {'Metric': 'True Skill Statistics', model_name: tss[0], baseline_name: tss_base[0]},
        {'Metric': 'Bias', model_name: bi[0], baseline_name: bi_base[0]}
    ]

    # Create formatted Dash table
    error_table = dash_table.DataTable(
        data=data,
        columns=[
            {'name': 'Metric', 'id': 'Metric'},
            {'name': model_name, 'id': model_name, 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)},
            {'name': baseline_name, 'id': baseline_name, 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)}
        ],
        style_cell_conditional=[
            {'if': {'column_id': model_name}, 'textAlign': 'right'},
            {'if': {'column_id': baseline_name}, 'textAlign': 'right'}
        ],
        **my_tablelayout
    )

    return error_table


def create_binary_outcome_bars(predicted_data, observed_data, baseline_data, threshold):
    """
    Creates a grouped bar chart comparing binary event classification 
    outcomes (Hit, False Alarm, Miss, Correct Rejection) for a model 
    prediction and a baseline (benchmark) prediction.

            Parameters:
                    predicted_data (pandas.DataFrame):
                        Model predictions. Must contain:
                            - 'Time' (datetime-like): timestamps
                            - 'Model': predicted values
                            - 'Name': model name (string, used for legend)
                    
                    observed_data (pandas.DataFrame):
                        Observational reference data. Must contain:
                            - 'Time' (datetime-like): timestamps
                            - 'Obs': observed values
                            - 'Name': dataset name
                    
                    baseline_data (pandas.DataFrame):
                        Baseline or benchmark predictions. Must contain:
                            - 'Time' (datetime-like): timestamps
                            - 'Model': predicted values
                            - 'Name': baseline name (string, used for legend)
                    
                    threshold(float):
                        Threshold used by `classifyBinaryEvents` to 
                        classify events as:
                            - Hit (true positive)
                            - Miss (false negative)
                            - False Alarm (false positive)
                            - Correct Rejection (true negative)
            
            Returns:
                    plotly.graph_objects.Figure:
                        Interactive grouped bar chart showing counts for each 
                        outcome type, comparing model vs. baseline.
    """

    baseline_as_obs = baseline_data[['Model']].rename(columns={'Model': 'Obs'})
    _, ct = classifyBinaryEvents(predicted_data, observed_data, threshold)
    _, ct_base = classifyBinaryEvents(predicted_data, baseline_as_obs, threshold)

    # Create the side-by-side bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['Hit', 'False Alarm', 'Miss', 'Correct Rejection'],
        y=ct.values.flatten(),
        marker=dict(opacity=0.8),
        name=predicted_data['Name'][0]
    ))

    fig.add_trace(go.Bar(
        x=['Hit', 'False Alarm', 'Miss', 'Correct Rejection'],
        y=ct_base.values.flatten(),
        marker=dict(opacity=0.8),
        name=baseline_data['Name'][0]
    ))

    # Update the figure with my_figlayout
    fig.update_layout(my_figlayout)

    fig.update_layout(
        title='Outcome Counts', 
        barmode='group',
        xaxis_title='Outcome', 
        legend=dict(x=0.1, y=1.0),
        yaxis_title='Count', 
        height=500  
    )

    return fig


def create_timeline_with_peaks(dates, observed_data, predicted_data, observed_name, predicted_name,
                               minimum_speed, maximum_dt,
                               x_label='Date', y_label='Bulk Speed [km s⁻¹]',
                               plot_title='Peak Comparison'):
    """
    Create a time series plot highlighting detected peaks in observed and predicted data
    for event-based validation.

    This visualization shows:
    - Observed and predicted time series
    - Automatically detected peaks in both datasets
    - Hits (matched peaks), misses, and false alarms
    - A horizontal threshold line indicating the event speed threshold

    Parameters:
        dates : pandas.Series or array-like
            Datetime values corresponding to each observation/prediction.
        observed_data : pandas.Series or array-like
            Observed data values (e.g., solar wind speed).
        predicted_data : pandas.Series or array-like
            Predicted data values from the model.
        observed_name : list-like or pandas.Series
            Dataset name(s) for the observed data (first element used as label).
        predicted_name : list-like or pandas.Series
            Dataset name(s) for the predicted data (first element used as label).
        minimum_speed : float
            Minimum speed threshold for peak detection.
        maximum_dt : float
            Maximum time difference (in hours) allowed when matching peaks between
            observations and predictions.
        x_label : str, optional
            X-axis label for the plot. Default is `'Date'`.
        y_label : str, optional
            Y-axis label for the plot. Default is `'Bulk Speed [km s⁻¹]'`.
        plot_title : str, optional
            Plot title. Default is `'Peak Comparison'`.

    Returns:
        plotly.graph_objects.Figure
            An interactive Plotly figure showing the timeline with detected peaks and event matches.
    """

    # Ensure dates is in datetime format
    dates = pd.to_datetime(dates)

    # Compute peaks and peak information
    peaks_obs, peaks_pred, hits, misses, false_alarms, time_diff, data_diff = compute_peaks(dates, observed_data, predicted_data, minimum_speed, maximum_dt)

    # Plot timeline for event-based validation using Line plot for observed and predicted data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=predicted_data, mode='lines', name=predicted_name))
    fig.add_trace(go.Scatter(x=dates, y=observed_data, mode='lines', name=observed_name, line=obs_linelayout))

    fig.add_trace(go.Scatter(x=dates.iloc[peaks_obs], y=observed_data.iloc[peaks_obs],
                             mode='markers', 
                             marker=dict(size=7, color='#1f77b4', symbol='circle', line=dict(width=1, color='black')),
                             name='Peaks in Observation'))
                             
    fig.add_trace(go.Scatter(x=dates.iloc[peaks_pred], y=predicted_data.iloc[peaks_pred],
                             mode='markers', 
                             marker=dict(size=7, color='#ff7f0e', symbol='circle', line=dict(width=1, color='black')),
                             name='Peaks in Prediction'))

    fig.add_trace(go.Scatter(x=dates.iloc[hits], y=20+predicted_data.iloc[hits],
                             mode='markers', 
                             marker=dict(size=10, color='green', symbol='star', line=dict(width=1, color='black')),
                             name='Hit'))

    fig.add_trace(go.Scatter(x=dates.iloc[misses], y=20+observed_data.iloc[misses],
                             mode='markers', 
                             marker=dict(size=10, color='red', symbol='x', line=dict(width=1, color='black')),
                             name='Miss'))

    fig.add_trace(go.Scatter(x=dates.iloc[false_alarms], y=20+predicted_data.iloc[false_alarms],
                             mode='markers', 
                             marker=dict(size=10, color='#9467bd', symbol='diamond', line=dict(width=1, color='black')),
                             name='False Alarm'))


    # Add a horizontal line at the minimum speed threshold
    fig.add_shape(type='line', x0=dates.min(), x1=dates.max(), y0=minimum_speed, y1=minimum_speed, line=dict(color='gray', width=2, dash='dash'), name='Minimum Speed Threshold')

    # Update the figure with my_figlayout
    fig.update_layout(my_figlayout)

    # Apply custom layout if defined
    fig.update_layout(
        title=plot_title, 
        xaxis_title=x_label, 
        yaxis_title=y_label, 
        xaxis=dict(range=[dates.iloc[0], dates.iloc[-1]]),
        legend=dict(orientation='h', x=0.075, y=1.05),
        height=400
    )

    return fig


def create_peak_bar_chart(time, obs, model,
                          threshold_value, maximum_dt,
                          baseline_time=None, baseline_model=None,
                          x_label='Outcome', y_label='Count', plot_title='Outcome Counts'):
    '''
    Detects peaks for a model (and optional baseline), computes hits/misses/false alarms,
    and creates a grouped bar chart comparing their performance.

            Parameters:
                    time (pandas.Series): Time values for the model
                    obs (pandas.Series): Observed data values
                    model (pandas.Series): Predicted model data values
                    threshold_value (float): Threshold for event detection
                    maximum_dt (float): Maximum time difference allowed for peak matching
                    baseline_time (pandas.Series, optional): Time values for baseline model
                    baseline_model (pandas.Series, optional): Predicted baseline model values
                    x_label (str): Label for the x-axis [default = 'Outcome']
                    y_label (str): Label for the y-axis [default = 'Count']
                    plot_title (str): Title for the plot [default = 'Outcome Counts']

            Returns:
                    plotly.graph_objects.Figure: Grouped bar chart figure
    '''

    # Compute peaks for model
    peaks_obs, peaks_pred, hits, misses, false_alarms, _, _ = compute_peaks(
        time, obs, model, threshold_value, maximum_dt
    )

    # Prepare counts for the model
    categories = ['Hit', 'Miss', 'False Alarm']
    model_counts = [len(hits), len(misses), len(false_alarms)]

    fig = go.Figure()

    # Add model bars
    fig.add_trace(go.Bar(
        x=categories,
        y=model_counts,
        name='Model',
        marker=dict(opacity=0.8)
    ))

    # Optional: compute peaks for baseline
    if baseline_time is not None and baseline_model is not None:
        _, _, hits_base, misses_base, false_alarms_base, _, _ = compute_peaks(
            baseline_time, obs, baseline_model, threshold_value, maximum_dt
        )
        baseline_counts = [len(hits_base), len(misses_base), len(false_alarms_base)]
        fig.add_trace(go.Bar(
            x=categories,
            y=baseline_counts,
            name='Baseline Model',
            marker=dict(opacity=0.8)
        ))

    # Layout
    fig.update_layout(
        my_figlayout,
        title=plot_title,
        barmode='group',
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        legend=dict(orientation='h', x=0.5, y=1.05)
    )

    return fig


def create_error_table_events(time, observed_data, model_data,
                              threshold_value, maximum_dt,
                              baseline_time=None, baseline_model=None,
                              model_name='Model', baseline_name='Baseline Model'):
    """
    Compute event-based binary classification metrics (from compute_peaks) and
    return a formatted Dash DataTable for model vs (optional) baseline.

            Parameters:
                    time (pandas.Series): Time values aligned with observed/model
                    observed_data (pandas.Series): Observed values
                    model_data (pandas.Series): Model values
                    threshold_value (float): Threshold used in compute_peaks
                    maximum_dt (float): Max allowed time diff for peak matching
                    baseline_time (pandas.Series, optional): Time for baseline model
                    baseline_model (pandas.Series, optional): Baseline model values
                    model_name (str): Column header for the model metrics
                    baseline_name (str): Column header for the baseline metrics

            Returns:
                    dash_table.DataTable: Metrics table (Model and optional Baseline columns)
    """
    # Contingency table entries
    _, _, hits, misses, false_alarms, _, _ = compute_peaks(
        time, observed_data, model_data, threshold_value, maximum_dt
    )
    ct = pd.DataFrame(
        [[len(hits), len(false_alarms), len(misses), 0]],
        columns=['tp', 'fp', 'fn', 'tn']
    )

    # Baseline contingency
    ct_base = None
    if baseline_time is not None and baseline_model is not None:
        _, _, hits_b, misses_b, false_alarms_b, _, _ = compute_peaks(
            baseline_time, observed_data, baseline_model, threshold_value, maximum_dt
        )
        ct_base = pd.DataFrame(
            [[len(hits_b), len(false_alarms_b), len(misses_b), 0]],
            columns=['tp', 'fp', 'fn', 'tn']
        )

    # Metrics for model
    tpr = truePositiveRate(ct)[0]
    fnr = falseNegativeRate(ct)[0]
    ppv = positivePredictiveValue(ct)[0]
    far = falseAlarmRatio(ct)[0]
    ts  = threatScore(ct)[0]
    bi  = bias(ct)[0]

    # Metrics for baseline (if provided)
    if ct_base is not None:
        tpr_b = truePositiveRate(ct_base)[0]
        fnr_b = falseNegativeRate(ct_base)[0]
        ppv_b = positivePredictiveValue(ct_base)[0]
        far_b = falseAlarmRatio(ct_base)[0]
        ts_b  = threatScore(ct_base)[0]
        bi_b  = bias(ct_base)[0]

    rows = [
        {'Metric': 'True Positive Rate',      model_name: tpr},
        {'Metric': 'False Negative Rate',     model_name: fnr},
        {'Metric': 'Positive Predictive Value', model_name: ppv},
        {'Metric': 'False Alarm Ratio',       model_name: far},
        {'Metric': 'Threat Score',            model_name: ts},
        {'Metric': 'Bias',                    model_name: bi}
    ]

    if ct_base is not None:
        rows[0][baseline_name] = tpr_b
        rows[1][baseline_name] = fnr_b
        rows[2][baseline_name] = ppv_b
        rows[3][baseline_name] = far_b
        rows[4][baseline_name] = ts_b
        rows[5][baseline_name] = bi_b

    columns = [
        {'name': 'Metric', 'id': 'Metric'},
        {'name': model_name, 'id': model_name, 'type': 'numeric',
         'format': Format(precision=2, scheme=Scheme.fixed)},
    ]
    if ct_base is not None:
        columns.append(
            {'name': baseline_name, 'id': baseline_name, 'type': 'numeric',
             'format': Format(precision=2, scheme=Scheme.fixed)}
        )

    error_table = dash_table.DataTable(
        data=rows,
        columns=columns,
        style_cell_conditional=[
            {'if': {'column_id': model_name}, 'textAlign': 'right'},
            {'if': {'column_id': baseline_name}, 'textAlign': 'right'} if ct_base is not None else {}
        ],
        **my_tablelayout
    )

    return error_table


def create_histogram_plot(time, obs, model, threshold_value, maximum_dt,
                          x_label='Value', y_label='Count', plot_title='Histogram Plot',
                          num_bins=20, show_zero_line=True, add_stats=True,
                          plot_type='Data Diff'):
    """
    Creates an enhanced histogram plot for given input data.

            Parameters:
                    time (pandas.Series): Time values for the data points
                    obs (pandas.Series): Observed data values
                    model (pandas.Series): Predicted model data values
                    threshold_value (float): Threshold for event detection
                    maximum_dt (float): Maximum time difference allowed for peak matching
                    x_label (str): Label for the x-axis
                    y_label (str): Label for the y-axis
                    plot_title (str): Title for the plot
                    num_bins (int): Number of histogram bins
                    show_zero_line (bool): Whether to draw a zero reference line
                    add_stats (bool): Whether to display statistics box
                    plot_type (str): Type of plot: 'Data Diff' or 'Time Diff'
                    
            Returns:
                    plotly.graph_objects.Figure: The histogram plot figure
    """

    fig = go.Figure()

    peaks_obs, peaks_pred, hits, misses, false_alarms, time_diff, data_diff = compute_peaks(
        time, obs, model, threshold_value, maximum_dt
    )

    if plot_type == 'Data Diff':
        input_data = data_diff
    else:
        input_data = time_diff

    # Histogram
    hist_data = np.histogram(input_data, bins=num_bins)
    max_y = max(hist_data[0])

    fig.add_trace(go.Histogram(
        x=input_data,
        nbinsx=num_bins,
        marker_color='black',
        opacity=0.5,
        name='Histogram',
        showlegend=False
    ))

    # Zero line
    if show_zero_line and np.min(input_data) < 0 < np.max(input_data):
        fig.add_trace(go.Scatter(
            x=[0, 0],
            y=[0, max_y * 100],
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            name='Zero Line',
            showlegend=False
        ))

    # Statistics box
    if add_stats:
        mean_val = np.mean(input_data)
        std_val = np.std(input_data)
        median_val = np.median(input_data)
        skew_val = skew(input_data)

        stats_text = (f"<b>Statistics</b><br>"
                      f"Mean: {mean_val:.2f}<br>"
                      f"Median: {median_val:.2f}<br>"
                      f"Std: {std_val:.2f}<br>"
                      f"Skew: {skew_val:.2f}")

        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.98, y=0.95,
            xanchor="right", yanchor="top",
            showarrow=False,
            font=dict(size=11, color="black"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="lightgrey",
            borderwidth=1,
            opacity=0.9
        )

    # Symmetric x-axis
    abs_max = max(abs(np.min(input_data)), abs(np.max(input_data)))

    fig.update_layout(
        my_figlayout,
        title=plot_title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        bargap=0.05,
        xaxis=dict(
            range=[-abs_max, abs_max],
            zeroline=False,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            range=[0, max_y*1.1],
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey'
        ),
        legend=dict(orientation='h', x=0.05, y=1.0)
    )

    return fig


def create_scatter_plot_for_hits(time, observed_data, predicted_data,
                                 threshold_value, maximum_dt,
                                 max_points=1000, min_points=50, opacity=0.5,
                                 x_label='Observed Bulk Speed [km s⁻¹]',
                                 y_label='Modeled Bulk Speed [km s⁻¹]',
                                 plot_title='Scatter Plot for Hits',
                                 bounds=(200, 800)):
    '''
    Detects peaks (using compute_peaks), extracts matched "hit" pairs, and
    creates a density+scatter plot of observed vs. predicted hit values.

            Parameters:
                    time (pandas.Series): Time values aligned with observed/predicted
                    observed_data (pandas.Series): Observed values
                    predicted_data (pandas.Series): Predicted/model values
                    threshold_value (float): Threshold used inside compute_peaks
                    maximum_dt (float): Max allowed time difference for peak matching
                    max_points (int): Max plotted points (downsampling applied if exceeded)
                    min_points (int): Minimum points required to show KDE contours
                    opacity (float): Marker opacity for scatter points
                    x_label (str): X-axis label
                    y_label (str): Y-axis label
                    plot_title (str): Plot title
                    bounds (tuple): (min_val, max_val) for both axes

            Returns:
                    plotly.graph_objects.Figure: Scatter/KDE figure for hits
    '''
    # Compute peaks
    (peaks_obs, peaks_pred,
     hits, misses, false_alarms,
     time_diff, data_diff) = compute_peaks(time, observed_data, predicted_data,
                                           threshold_value, maximum_dt)

    # If there are no hits, return an empty figure
    if len(hits) == 0:
        min_val, max_val = bounds
        fig = go.Figure()
        fig.update_layout(
            my_figlayout,
            title=plot_title + ' (no hits found)',
            xaxis_title=x_label,
            yaxis_title=y_label,
            xaxis=dict(range=[min_val, max_val], showgrid=True, gridwidth=0.5, gridcolor='lightgrey'),
            yaxis=dict(range=[min_val, max_val], showgrid=True, gridwidth=0.5, gridcolor='lightgrey'),
            height=500,
            legend=dict(orientation='h', x=0.05, y=1.0)
        )
        return fig

    if isinstance(hits[0], (pd.Timestamp, np.datetime64)):
        idxer = time.index.get_indexer(hits)
        hit_indices = [i for i in idxer if i != -1]
    else:
        n = len(time)
        hit_indices = [i for i in hits if 0 <= int(i) < n]

    hit_obs = observed_data.iloc[hit_indices]
    hit_pred = predicted_data.iloc[hit_indices]

    data = pd.DataFrame({'Observed': hit_obs, 'Predicted': hit_pred}).dropna()

    # Downsample if needed
    if len(data) > max_points:
        step = max(1, len(data) // max_points)
        scatter_data = data.iloc[::step].head(max_points)
        downsampled_label = ' (downsampled)'
    else:
        scatter_data = data
        downsampled_label = ''

    min_val, max_val = bounds

    fig = go.Figure()

    # KDE contours
    if len(data) > min_points:
        values = np.vstack([data['Observed'].values, data['Predicted'].values])
        kde = gaussian_kde(values)
        xi, yi = np.meshgrid(
            np.linspace(min_val, max_val, 100),
            np.linspace(min_val, max_val, 100)
        )
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

        colorscale = [
            [0.0, 'rgba(255,255,255,0)'],
            [0.1, 'rgba(255,255,255,0)'],
            [0.3, 'rgba(255,200,100,0.2)'],
            [0.7, 'rgba(255,140,50,0.5)'],
            [1.0, 'rgba(255,100,0,0.8)']
        ]
        fig.add_trace(go.Contour(
            x=xi[0],
            y=yi[:, 0],
            z=zi,
            showscale=False,
            colorscale=colorscale,
            hoverinfo='skip',
            contours=dict(showlines=True),
            line=dict(width=0.5, color='lightgrey')
        ))

    # Scatter points
    fig.add_trace(go.Scatter(
        x=scatter_data['Observed'],
        y=scatter_data['Predicted'],
        mode='markers',
        marker=dict(size=5, color='black', opacity=opacity),
        name='Hits' + downsampled_label,
        showlegend=True
    ))

    # Diagonal y=x line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='grey', dash='dash', width=1),
        name='Diagonal',
        showlegend=False
    ))

    # Regression
    if len(data) >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(data['Observed'], data['Predicted'])
        r_squared = r_value ** 2
        reg_x = np.linspace(min_val, max_val, 100)
        reg_y = intercept + slope * reg_x

        fig.add_trace(go.Scatter(
            x=reg_x,
            y=reg_y,
            mode='lines',
            line=dict(color='black', dash='dashdot', width=1.5),
            name='Linear Regression',
            visible='legendonly'
        ))

        fig.add_annotation(
            text=f"<b>Linear Regression:</b><br>Slope: {slope:.2f}<br>Intercept: {intercept:.2f}<br>R²: {r_squared:.2f}",
            xref="paper", yref="paper",
            x=0.98, y=0.03,
            xanchor="right", yanchor="bottom",
            showarrow=False,
            font=dict(size=11, color="black"),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="lightgrey",
            borderwidth=1,
            opacity=0.8
        )

    # Layout
    fig.update_layout(
        my_figlayout,
        title=plot_title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(
            range=[min_val, max_val],
            tick0=min_val,
            dtick=100,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            range=[min_val, max_val],
            tick0=min_val,
            dtick=100,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey'
        ),
        height=500,
        legend=dict(orientation='h', x=0.05, y=1.0)
    )

    return fig


def create_error_scatter_plot(time, observed_data, predicted_data,
                              threshold_value, maximum_dt,
                              x_label='Time Error [hours]', y_label='Error [km s⁻¹]',
                              plot_title='Error Plot for Hits',
                              max_points=1000, min_points=50, opacity=0.5,
                              time_bounds=(-96, 96), data_bounds=(-400, 400)):
    '''
    Runs compute_peaks to find matched "hit" events, then plots the per-hit
    time error vs data error, with optional KDE contours and helpful guides.

            Parameters:
                    time (pandas.Series): Time values aligned with observed/predicted
                    observed_data (pandas.Series): Observed values
                    predicted_data (pandas.Series): Predicted/model values
                    threshold_value (float): Threshold used inside compute_peaks
                    maximum_dt (float): Max allowed time difference for peak matching (same units as time)
                    x_label (str): X-axis label [default = 'Time Error [hours]']
                    y_label (str): Y-axis label [default = 'Error [km s⁻¹]']
                    plot_title (str): Plot title
                    max_points (int): Max number of scatter points (downsample if exceeded)
                    min_points (int): Min points required to draw KDE contours
                    opacity (float): Marker opacity
                    time_bounds (tuple): (min,max) range for time error axis
                    data_bounds (tuple): (min,max) range for data error axis

            Returns:
                    plotly.graph_objects.Figure: Error scatter/KDE plot for hits
    '''
    # Compute peaks
    (_, _, hits, _, _, time_diff, data_diff) = compute_peaks(
        time, observed_data, predicted_data, threshold_value, maximum_dt
    )

    # Handle case with zero hits
    if len(hits) == 0:
        fig = go.Figure()
        fig.update_layout(
            my_figlayout,
            title=plot_title + ' (no hits found)',
            xaxis_title=x_label,
            yaxis_title=y_label,
            xaxis=dict(range=list(time_bounds), tick0=0, dtick=24, showgrid=True, gridwidth=0.5, gridcolor='lightgrey'),
            yaxis=dict(range=list(data_bounds), tick0=0, dtick=100, showgrid=True, gridwidth=0.5, gridcolor='lightgrey'),
            height=500,
            legend=dict(orientation='h', x=0.05, y=1.0)
        )
        # zero reference lines
        fig.add_trace(go.Scatter(x=list(time_bounds), y=[0, 0], mode='lines',
                                 line=dict(color='gray', dash='dash', width=1), showlegend=False))
        fig.add_trace(go.Scatter(x=[0, 0], y=list(data_bounds), mode='lines',
                                 line=dict(color='gray', dash='dash', width=1), showlegend=False))
        return fig

    # Build hover labels
    try:
        if isinstance(hits[0], (pd.Timestamp, np.datetime64)):
            hover_dates = [str(h) for h in hits]
        else:
            hover_dates = [str(time.iloc[int(h)]) for h in hits]
    except Exception:
        hover_dates = [str(h) for h in hits]

    data = pd.DataFrame({
        'Time Error': np.asarray(time_diff),
        'Data Error': np.asarray(data_diff),
        'Date': hover_dates
    }).dropna()

    # Downsample if needed
    if len(data) > max_points:
        step = max(1, len(data) // max_points)
        scatter_data = data.iloc[::step].head(max_points)
        label_suffix = ' (downsampled)'
    else:
        scatter_data = data
        label_suffix = ''

    # Prepare figure
    fig = go.Figure()

    # KDE contour if enough data
    if len(data) > min_points:
        values = np.vstack([data['Time Error'].values, data['Data Error'].values])
        kde = gaussian_kde(values)

        # Add padding to avoid cutoff
        pad_x = (data['Time Error'].max() - data['Time Error'].min()) * 0.25 or 1.0
        pad_y = (data['Data Error'].max() - data['Data Error'].min()) * 0.25 or 1.0

        xi, yi = np.meshgrid(
            np.linspace(data['Time Error'].min() - pad_x, data['Time Error'].max() + pad_x, 100),
            np.linspace(data['Data Error'].min() - pad_y, data['Data Error'].max() + pad_y, 100)
        )
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

        colorscale = [
            [0.0, 'rgba(255,255,255,0)'],
            [0.1, 'rgba(255,255,255,0)'],
            [0.3, 'rgba(255,200,100,0.2)'],
            [0.7, 'rgba(255,140,50,0.5)'],
            [1.0, 'rgba(255,100,0,0.8)']
        ]
        fig.add_trace(go.Contour(
            x=xi[0],
            y=yi[:, 0],
            z=zi,
            showscale=False,
            colorscale=colorscale,
            hoverinfo='skip',
            contours=dict(showlines=True),
            line=dict(width=0.5, color='lightgrey')
        ))

    # Scatter points
    fig.add_trace(go.Scatter(
        x=scatter_data['Time Error'],
        y=scatter_data['Data Error'],
        mode='markers',
        marker=dict(size=5, color='black', opacity=opacity),
        name='Error Points' + label_suffix,
        text=scatter_data['Date'],
        hovertemplate="<b>Date:</b> %{text}"
                      "<br><b>Time Error [h]:</b> %{x}"
                      "<br><b>Error [km s⁻¹]:</b> %{y}<extra></extra>",
        showlegend=False
    ))

    # Zero reference lines
    fig.add_trace(go.Scatter(
        x=list(time_bounds),
        y=[0, 0],
        mode='lines',
        line=dict(color='gray', dash='dash', width=1),
        name='Zero Data Error',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=list(data_bounds),
        mode='lines',
        line=dict(color='gray', dash='dash', width=1),
        name='Zero Time Error',
        showlegend=False
    ))

    # Quadrant annotations
    annotations = [
        dict(x=(time_bounds[0] + time_bounds[1]) * 0.25, y=data_bounds[1] * 0.9,
             text="Overestimated peaks", showarrow=False,
             font=dict(size=11, color='grey'), align="left"),
        dict(x=(time_bounds[0] + time_bounds[1]) * 0.25, y=data_bounds[0] * 0.9,
             text="Underestimated peaks", showarrow=False,
             font=dict(size=11, color='grey'), align="left")
    ]

    # Layout
    fig.update_layout(
        my_figlayout,
        title=plot_title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(
            range=list(time_bounds),
            tick0=0,
            dtick=24,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            range=list(data_bounds),
            tick0=0,
            dtick=100,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey'
        ),
        height=500,
        legend=dict(orientation='h', x=0.05, y=1.0),
        annotations=annotations
    )

    return fig


def create_time_error_vs_observed_data_plot(time, observed_data, predicted_data,
                                            threshold_value, maximum_dt,
                                            x_label='Time Error [hours]',
                                            y_label='Observed Bulk Speed [km s⁻¹]',
                                            plot_title='Time Error vs. Observed Bulk Speed',
                                            max_points=1000, min_points=50, opacity=0.5,
                                            time_bounds=(-96, 96), obs_bounds=(200, 800)):
    '''
    Runs compute_peaks, then plots per-hit time error vs observed peak value,
    with optional KDE contours, zero line, and quadrant annotations.

            Parameters:
                    time (pandas.Series): Time values aligned with observed/predicted
                    observed_data (pandas.Series): Observed values
                    predicted_data (pandas.Series): Predicted/model values
                    threshold_value (float): Threshold used inside compute_peaks
                    maximum_dt (float): Max allowed time difference for peak matching
                    x_label (str): X-axis label [default = 'Time Error [hours]']
                    y_label (str): Y-axis label [default = 'Observed Bulk Speed [km s⁻¹]']
                    plot_title (str): Plot title
                    max_points (int): Max number of scatter points (downsample if exceeded)
                    min_points (int): Min points required to draw KDE contours
                    opacity (float): Marker opacity
                    time_bounds (tuple): (min,max) range for time error axis
                    obs_bounds (tuple): (min,max) range for observed axis

            Returns:
                    plotly.graph_objects.Figure: Scatter/KDE plot
    '''
    # Compute peaks
    (peaks_obs, peaks_pred,
     hits, misses, false_alarms,
     time_diff, data_diff) = compute_peaks(time, observed_data, predicted_data,
                                           threshold_value, maximum_dt)

    # Handle no hits
    if len(hits) == 0:
        fig = go.Figure()
        fig.update_layout(
            my_figlayout,
            title=plot_title + ' (no hits found)',
            xaxis_title=x_label,
            yaxis_title=y_label,
            xaxis=dict(range=list(time_bounds), tick0=0, dtick=24, showgrid=True, gridwidth=0.5, gridcolor='lightgrey'),
            yaxis=dict(range=list(obs_bounds), tick0=obs_bounds[0], dtick=100, showgrid=True, gridwidth=0.5, gridcolor='lightgrey'),
            height=500,
            legend=dict(orientation='h', x=0.05, y=1.0)
        )
        # zero time-error line
        fig.add_trace(go.Scatter(x=[0, 0], y=list(obs_bounds), mode='lines',
                                 line=dict(color='gray', dash='dash', width=1), showlegend=False))
        return fig

    if isinstance(hits[0], (pd.Timestamp, np.datetime64)):
        idxer = time.index.get_indexer(hits)
        hit_indices = [i for i in idxer if i != -1]
        hover_dates = [str(h) for h in hits]
    else:
        n = len(time)
        hit_indices = [i for i in hits if 0 <= int(i) < n]
        hover_dates = [str(time.iloc[int(h)]) for h in hit_indices]

    # Observed values at hit positions
    hits_observed_data = observed_data.iloc[hit_indices]

    # Assemble frame (assumes time_diff aligns 1:1 with hits)
    data = pd.DataFrame({
        'Time Error': np.asarray(time_diff),
        'Observed Data': np.asarray(hits_observed_data),
        'Date': hover_dates[:len(hit_indices)]
    }).dropna()

    # Downsample
    if len(data) > max_points:
        step = max(1, len(data) // max_points)
        scatter_data = data.iloc[::step].head(max_points)
        label_suffix = ' (downsampled)'
    else:
        scatter_data = data
        label_suffix = ''

    fig = go.Figure()

    # KDE contour if enough data
    if len(data) > min_points:
        values = np.vstack([data['Time Error'].values, data['Observed Data'].values])
        kde = gaussian_kde(values)

        pad_x = (data['Time Error'].max() - data['Time Error'].min()) * 0.25 or 1.0
        xi, yi = np.meshgrid(
            np.linspace(data['Time Error'].min() - pad_x, data['Time Error'].max() + pad_x, 100),
            np.linspace(obs_bounds[0], obs_bounds[1], 100)
        )
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

        colorscale = [
            [0.0, 'rgba(255,255,255,0)'],
            [0.1, 'rgba(255,255,255,0)'],
            [0.3, 'rgba(255,200,100,0.2)'],
            [0.7, 'rgba(255,140,50,0.5)'],
            [1.0, 'rgba(255,100,0,0.8)']
        ]
        fig.add_trace(go.Contour(
            x=xi[0],
            y=yi[:, 0],
            z=zi,
            showscale=False,
            colorscale=colorscale,
            hoverinfo='skip',
            contours=dict(showlines=True),
            line=dict(width=0.5, color='lightgrey')
        ))

    # Scatter points
    fig.add_trace(go.Scatter(
        x=scatter_data['Time Error'],
        y=scatter_data['Observed Data'],
        mode='markers',
        marker=dict(size=5, color='black', opacity=opacity),
        name='Hit Points' + label_suffix,
        text=scatter_data['Date'],
        hovertemplate="<b>Date:</b> %{text}"
                      "<br><b>Time Error [h]:</b> %{x}"
                      "<br><b>Observed [km s⁻¹]:</b> %{y}<extra></extra>",
        showlegend=False
    ))

    # Vertical zero line
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=list(obs_bounds),
        mode='lines',
        line=dict(color='gray', dash='dash', width=1),
        name='Zero Time Error',
        showlegend=False
    ))

    # Quadrant annotations (positions scaled to bounds)
    xq = (time_bounds[1] - time_bounds[0]) * 0.76 / 2   # ~±73 for default bounds
    annotations = [
        dict(x=-xq, y=obs_bounds[1] * 0.96, text="Late predictions<br>High peak values", showarrow=False,
             font=dict(size=11, color='grey'), align="left"),
        dict(x= xq, y=obs_bounds[1] * 0.96, text="Early predictions<br>High peak values", showarrow=False,
             font=dict(size=11, color='grey'), align="right"),
        dict(x=-xq, y=obs_bounds[0] + (obs_bounds[1]-obs_bounds[0]) * 0.15, text="Late predictions<br>Low peak values", showarrow=False,
             font=dict(size=11, color='grey'), align="left"),
        dict(x= xq, y=obs_bounds[0] + (obs_bounds[1]-obs_bounds[0]) * 0.15, text="Early predictions<br>Low peak values", showarrow=False,
             font=dict(size=11, color='grey'), align="right")
    ]

    # Layout
    fig.update_layout(
        my_figlayout,
        title=plot_title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(
            range=list(time_bounds),
            tick0=0,
            dtick=24,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            range=list(obs_bounds),
            tick0=obs_bounds[0],
            dtick=100,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey'
        ),
        height=500,
        legend=dict(orientation='h', x=0.05, y=1.0),
        annotations=annotations
    )

    return fig


def create_mm_line_chart(predicted_data_list, observed_data):
    """
    Create a multi-model time series line chart comparing predicted bulk speeds 
    from multiple models against observed data.

    Parameters:
        predicted_data_list : list of pandas.DataFrame
            A list where each DataFrame contains the predicted time series for one 
            model. Each DataFrame must include:
                - 'Time'  (datetime64) : Time stamps for the predictions.
                - 'Model' (float)      : Predicted bulk speed values [km s⁻¹].
                - 'Name'  (str)        : Model display name (same value for all rows).
            All 'Time' columns should use the same timezone convention (UTC recommended).

        observed_data : pandas.DataFrame
            DataFrame containing the observed data with:
                - 'Time' (datetime64)  : Observation timestamps.
                - 'Obs'  (float)       : Observed bulk speed values [km s⁻¹].
                - 'Name' (str)         : Observations display name.

    Returns:
        plotly.graph_objects.Figure
            A Plotly Figure object containing the line chart. 
    """

    fig = go.Figure()

    # Define a list of colors
    colors = px.colors.qualitative.Plotly

    # Add traces for each predicted model
    for i, predicted_data in enumerate(predicted_data_list):
        fig.add_trace(go.Scatter(
            x=predicted_data["Time"],
            y=predicted_data["Model"],
            mode="lines",
            line=dict(color=colors[i % len(colors)], width=2),
            name=predicted_data["Name"].iloc[0],
            hovertemplate="Time=%{x}<br>Model=%{y:.2f}<extra></extra>",
        ))

    # Add observed data trace
    fig.add_trace(go.Scatter(
        x=observed_data["Time"],
        y=observed_data["Obs"],
        name=observed_data["Name"].iloc[0],
        mode="markers",
        marker=dict(
            symbol="circle-open",
            size=2,
            color="#222222",
            line=dict(width=1),
        ),
        opacity=0.85,
        hovertemplate="Time=%{x}<br>Obs=%{y:.2f}<extra></extra>",
    ))
    x_start = pd.to_datetime(observed_data["Time"].iloc[0])
    x_end   = pd.to_datetime(observed_data["Time"].iloc[-1])

    # Update layout
    fig.update_layout(
        my_figlayout,
        title="Time Series Comparison",
        xaxis_title="Date",
        yaxis_title="Bulk Speed [km s⁻¹]",
        legend=dict(orientation="h", x=0.03, y=1.05),
        xaxis=dict(range=[x_start, x_end]), 
        height=400,
    )

    return fig


def create_mm_box_plot(aligned_predicted_data_list, aligned_observed_data):
    """
    Create a comparative box plot showing the statistical distribution of predicted
    bulk speeds from multiple models alongside observed data.

    Parameters:
        aligned_predicted_data_list : list of pandas.DataFrame
            A list where each DataFrame contains time-aligned predicted bulk speed data 
            for one model, with:
                - 'Time'  (datetime64) : Timestamps of predictions (alignment already done).
                - 'Model' (float)      : Predicted bulk speed values [km s⁻¹].
                - 'Name'  (str)        : Model display name (identical for all rows in that DF).

        aligned_observed_data : pandas.DataFrame
            DataFrame containing the observed bulk speed data, with:
                - 'Time' (datetime64)  : Observation timestamps (aligned to models).
                - 'Obs'  (float)       : Observed bulk speed values [km s⁻¹].
                - 'Name' (str)         : Observations display name.

    Returns:
        plotly.graph_objects.Figure
            A Plotly Figure object containing the box plots.
    """

    combined_data = []

    # Add predicted data
    for aligned_predicted_data in aligned_predicted_data_list:
        df = aligned_predicted_data.rename(columns={'Model': 'Value'})
        df['Type'] = aligned_predicted_data['Name'][0]
        df['Color'] = 'default'
        combined_data.append(df)

    # Add observed data
    observed_name = aligned_observed_data['Name'][0]
    obs_df = aligned_observed_data.rename(columns={'Obs': 'Value'})
    obs_df['Type'] = observed_name
    obs_df['Color'] = 'gray'
    combined_data.append(obs_df)

    # Combine all
    combined_data = pd.concat(combined_data, ignore_index=True)

    fig = go.Figure()

    types = combined_data['Type'].unique()
    for t in types:
        subset = combined_data[combined_data['Type'] == t]
        color = 'gray' if t == observed_name else None

        fig.add_trace(go.Box(
            y=subset['Value'],
            name=t,
            marker_color=color if color else None,
            boxmean=True, 
            boxpoints=False,
        ))

    # Layout updates
    fig.update_layout(
        my_figlayout,
        title='Statistical Distribution',
        yaxis_title='Bulk Speed [km s⁻¹]',
        xaxis_title='Category',
        legend=dict(orientation='h', x=0.03, y=1.05),
        legend_title='',
        height=360
    )

    return fig


def create_mm_rmse_timeline(predicted_data_list, observed_data, benchmark_data,normalize_time="naive_utc", floor="H"):
    """
    Creates a multi-model RMSE timeline across Carrington Rotations,
    including optional relative skill vs a benchmark.

    Parameters:
        predicted_data_list : list[pd.DataFrame]
            Each DF must contain: 'Time' (datetime-like), 'Model' (float), 'Name' (str)
        observed_data : pd.DataFrame
            Must contain: 'Time' (datetime-like), 'Obs' (float), 'Name' (str)
        benchmark_data : pd.DataFrame
            Must contain: 'Time' (datetime-like), 'Model' (float), 'Name' (str)
        normalize_time : {"naive_utc", "aware_utc"}, default "naive_utc"
            - "naive_utc": make timestamps tz-naive in UTC (consistent with earlier notebook code)
            - "aware_utc": make timestamps tz-aware (UTC)
        floor : str or None, default "H"
            Pandas offset alias to floor timestamps (e.g., "H", "30min"). Use None to skip.

    Returns:
        plotly.graph_objects.Figure
            Figure with Absolute RMSE (default) and toggle for Relative Skill vs benchmark.
    """

    def _norm_time_col(df, time_col="Time"):
        d = df.copy()
        d[time_col] = pd.to_datetime(d[time_col], errors="coerce", utc=True)
        if normalize_time == "naive_utc":
            d[time_col] = d[time_col].dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            d[time_col] = d[time_col].dt.tz_convert("UTC")
        if floor:
            d[time_col] = d[time_col].dt.floor(floor)
        return d

    # Normalize all inputs
    observed_data   = _norm_time_col(observed_data, "Time")
    benchmark_data  = _norm_time_col(benchmark_data, "Time")
    predicted_data_list = [_norm_time_col(df, "Time") for df in predicted_data_list]

    # Names
    model_names = []
    for df in predicted_data_list:
        try:
            model_names.append(str(df["Name"].iloc[0]))
        except Exception:
            model_names.append("Model")
    try:
        benchmark_name = str(benchmark_data["Name"].iloc[0])
    except Exception:
        benchmark_name = "Benchmark"

    def _cr_bounds(cr):
        start = carrington_rotation_time(cr, 360 * u.deg).datetime
        end   = carrington_rotation_time(cr, 1 * u.deg).datetime
        if normalize_time == "naive_utc":
            return pd.Timestamp(start), pd.Timestamp(end)
        else:
            return pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")

    epoch = pd.Timestamp("1853-11-09")
    if normalize_time != "naive_utc":
        epoch = epoch.tz_localize("UTC")
    start_time = observed_data["Time"].min()
    end_time   = observed_data["Time"].max()

    if pd.isna(start_time) or pd.isna(end_time):
        fig = go.Figure()
        fig.update_layout(title="RMSE Comparison Over Time (no data)")
        return fig

    cr_len_days = 27.2753
    cr_start = 1 + int((start_time - epoch) / pd.Timedelta(days=cr_len_days))
    cr_end   = int((end_time - epoch)   / pd.Timedelta(days=cr_len_days))

    cr_numbers = []
    rmse_values = {name: [] for name in model_names}
    rel_skill   = {name: [] for name in model_names}
    benchmark_rmse = []

    # Compute per-CR RMSEs
    for cr in range(cr_start, cr_end + 1):
        cr_start_time, cr_end_time = _cr_bounds(cr)

        obs_cr   = observed_data[(observed_data["Time"] >= cr_start_time) & (observed_data["Time"] < cr_end_time)]
        bench_cr = benchmark_data[(benchmark_data["Time"]  >= cr_start_time) & (benchmark_data["Time"]  < cr_end_time)]

        if obs_cr.empty or bench_cr.empty:
            continue

        # Merge benchmark vs obs on Time
        merged_bench = pd.merge(bench_cr[["Time","Model"]],
                                obs_cr[["Time","Obs"]],
                                on="Time", how="inner")
        if merged_bench.empty:
            continue

        # Compute benchmark RMSE
        # rmsScore expects DF with identical indices/columns; align via column rename
        rmse_b = rmsScore(
            merged_bench[["Model"]],
            merged_bench[["Obs"]].rename(columns={"Obs": "Model"})
        ).iloc[0]
        benchmark_rmse.append(rmse_b)

        for df, model_name in zip(predicted_data_list, model_names):
            model_cr = df[(df["Time"] >= cr_start_time) & (df["Time"] < cr_end_time)]
            merged = pd.merge(model_cr[["Time","Model"]],
                              obs_cr[["Time","Obs"]],
                              on="Time", how="inner")
            if merged.empty:
                rmse_values[model_name].append(np.nan)
                rel_skill[model_name].append(np.nan)
                continue

            rmse_m = rmsScore(
                merged[["Model"]],
                merged[["Obs"]].rename(columns={"Obs": "Model"})
            ).iloc[0]

            rmse_values[model_name].append(rmse_m)
            rel_skill[model_name].append(100 * (rmse_b - rmse_m) / rmse_b if rmse_b else np.nan)

        cr_numbers.append(cr)

    fig = go.Figure()

    # Absolute RMSE traces
    for model_name in model_names:
        fig.add_trace(go.Scatter(
            x=cr_numbers, y=rmse_values[model_name],
            mode='lines+markers', name=model_name
        ))

    # Benchmark trace
    fig.add_trace(go.Scatter(
        x=cr_numbers, y=benchmark_rmse,
        mode='lines+markers', name=benchmark_name,
        line=dict(dash='dash', color='grey')
    ))

    # Relative skill traces
    for model_name in model_names:
        fig.add_trace(go.Scatter(
            x=cr_numbers, y=rel_skill[model_name],
            mode='lines+markers', name=f"{benchmark_name} vs. {model_name}",
            line=dict(dash='dot'),
            visible=False
        ))

    # Zero line for relative skill
    fig.add_trace(go.Scatter(
        x=cr_numbers, y=[0] * len(cr_numbers),
        mode='lines+markers', line=dict(color='grey', dash='dash'),
        name='Benchmark Reference', visible=False, showlegend = False
    ))

    # Dropdown menus
    num_models = len(model_names)
    num_traces = len(fig.data)

    abs_vis = [True] * (num_models + 1) + [False] * (num_models + 1)
    rel_vis = [False] * (num_models + 1) + [True] * num_models + [True]

    # Annotation for Relative Skill
    annotation_explainer = dict(
        text="Above zero: better than benchmark<br>Below zero: worse than benchmark",
        xref="paper", yref="paper",
        x=0.98, y=0.2,
        xanchor="right", yanchor="top",
        showarrow=False,
        font=dict(size=11, color="black"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="lightgrey",
        borderwidth=1,
        opacity=0.9
    )

    # Add annotation
    fig.update_layout(annotations=[{**annotation_explainer, "visible": False}])

    fig.update_layout(
        my_figlayout,
        title="RMSE Comparison Over Time",
        xaxis_title="Carrington Rotation Number",
        yaxis_title="RMSE [km s⁻¹]",
        height=500,
        xaxis=dict(
            range=[cr_start-0.25, cr_end+0.25],
            tickmode='linear',
            tick0=cr_numbers[0],
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            dtick = 50, 
            tick0 = 0,
            showgrid = True,
            gridwidth = 0.5,
            gridcolor = 'lightgrey'
        ),
        legend=dict(
            orientation='h',
            x=0,
            y=-0.25,
            xanchor='left',
            yanchor='top',
            traceorder='normal',
            title_text=''
        ),
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="Absolute Values",
                        method="update",
                        args=[
                            {"visible": abs_vis},
                            {
                                "yaxis": {
                                    "title": "RMSE [km s⁻¹]",
                                    "tick0": 0,
                                    "dtick": 50,
                                    "showgrid": True,
                                    "gridwidth": 0.5,
                                    "gridcolor": 'lightgrey'
                                },
                                "xaxis": {
                                    "range": [cr_start-0.25, cr_end+0.25],
                                    "title": "Carrington Rotation Number",
                                    "tickmode": "linear",
                                    "tick0": cr_numbers[0],
                                    "showgrid": True,
                                    "gridwidth": 0.5,
                                    "gridcolor": "lightgrey"
                                },
                                "annotations": [{**annotation_explainer, "visible": False}]
                            }
                        ]
                    ),
                    dict(
                        label="Relative Skill [%]",
                        method="update",
                        args=[
                            {"visible": rel_vis},
                            {
                                "yaxis": {
                                    "title": "Relative Skill [%]",
                                    "tick0": 0,
                                    "showgrid": True,
                                    "gridwidth": 0.5,
                                    "gridcolor": 'lightgrey'
                                },
                                "xaxis": {
                                    "range": [cr_start-0.25, cr_end+0.25],
                                    "title": "Carrington Rotation Number",
                                    "tickmode": "linear",
                                    "tick0": cr_numbers[0],
                                    "showgrid": True,
                                    "gridwidth": 0.5,
                                    "gridcolor": "lightgrey"
                                },
                                "annotations": [{**annotation_explainer, "visible": True}]
                            }
                        ]
                    )
                ],
                direction="down",
                showactive=True,
                x=0.26,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ]
    )
    return fig


def create_mm_error_table_ptp(observed_data, predicted_data_list):
    """
    Build a point‑to‑point error summary table comparing multiple models
    against observed bulk speed time series. This function computes common 
    continuous‑variable metrics for each model and returns a formatted Dash 
    DataTable suitable for display in the dashboard.

    Metrics:
    • Mean Error (ME) [km s⁻¹]
    • Mean Absolute Error (MAE) [km s⁻¹]
    • Mean Squared Error (MSE) [(km s⁻¹)²]
    • Root Mean Squared Error (RMSE) [km s⁻¹]
    • Correlation Coefficient (CC) [unitless]
    • Median Symmetric Accuracy (MSA) [%]
    • Symmetric Signed Percentage Bias (SSPB) [%]

    Parameters:
        observed_data : pandas.DataFrame
            Observation frame with columns:
                - 'Time' (datetime64): timestamps
                - 'Obs'  (float)     : observed bulk speed [km s⁻¹]
                - 'Name' (str)       : label for observations (single value repeated)
            Note: Timestamps should already be normalized (same tz/naivety) and,
            if needed, floored to the desired cadence before calling.

        predicted_data_list : list[pandas.DataFrame]
            List of model frames; each must contain:
                - 'Time'  (datetime64): timestamps (aligned with observations)
                - 'Model' (float)     : predicted bulk speed [km s⁻¹]
                - 'Name'  (str)       : model display name (single value repeated)
            All frames must be aligned to the same time grid as `observed_data`
            (e.g., via `align_and_interpolate`), so point‑to‑point metrics make sense.

    Returns:
        dash_table.DataTable
            A Dash DataTable with one row per metric and one column per model.
            Numeric cells are right‑aligned and formatted to two decimals, using the
            global `my_tablelayout` styling.
    """

    observed_values = observed_data['Obs']

    metrics = [
        'Mean Error [km s⁻¹]',
        'Mean Absolute Error [km s⁻¹]',
        'Mean Squared Error [(km s⁻¹)²]',
        'Root Mean Squared Error [km s⁻¹]',
        'Correlation Coefficient',
        'Median Symmetric Accuracy [%]',
        'Symmetric Signed Percentage Bias [%]'
    ]

    model_names = [predicted_data['Name'][0] for predicted_data in predicted_data_list]
    error_metrics = {metric: [] for metric in metrics}

    for predicted_data in predicted_data_list:
        model_values = predicted_data['Model']
        error_metrics['Mean Error [km s⁻¹]'].append(round(meScore(model_values, observed_values), 2))
        error_metrics['Mean Absolute Error [km s⁻¹]'].append(round(maeScore(model_values, observed_values), 2))
        error_metrics['Mean Squared Error [(km s⁻¹)²]'].append(round(meanSquaredError(model_values, observed_values), 2))
        error_metrics['Root Mean Squared Error [km s⁻¹]'].append(round(rmsScore(model_values, observed_values), 2))
        error_metrics['Correlation Coefficient'].append(round(ccScore(model_values, observed_values), 2))
        error_metrics['Median Symmetric Accuracy [%]'].append(round(msaScore(model_values, observed_values), 2))
        error_metrics['Symmetric Signed Percentage Bias [%]'].append(round(sspbScore(model_values, observed_values), 2))

    # Format table data
    data = [
        {'Metric': metric, **{model: error_metrics[metric][i] for i, model in enumerate(model_names)}}
        for metric in metrics
    ]

    columns = [{'name': 'Metric', 'id': 'Metric'}] + [
        {'name': model, 'id': model, 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)}
        for model in model_names
    ]

    style_cell_conditional = [
        {'if': {'column_id': model}, 'textAlign': 'right'} for model in model_names
    ]

    error_table = dash_table.DataTable(
        data=data,
        columns=columns,
        style_cell_conditional=style_cell_conditional,
        **my_tablelayout
    )

    return error_table


def create_mm_statistics_table(observed_data, predicted_data_list):
    """
    Build an extended statistics summary table for observed and predicted
    bulk speed time series.

    The function applies the `descriptiveStats` helper to the observation
    series and to each model series, then formats the results into a Dash
    DataTable for side-by-side comparison.

    Statistics:
    • Mean
    • Median
    • Standard Deviation (SD)
    • Minimum
    • Maximum
    • Variance
    • Skewness
    • Kurtosis

    Parameters:
        observed_data : pandas.DataFrame
            Observation frame with columns:
                - 'Time' (datetime64): timestamps
                - 'Obs'  (float)     : observed bulk speed [km s⁻¹]
                - 'Name' (str)       : label for observations (single value repeated)
            Timestamps should be normalized and aligned to match predictions.

        predicted_data_list : list[pandas.DataFrame]
            List of model frames; each must contain:
                - 'Time'  (datetime64): timestamps (aligned with observations)
                - 'Model' (float)     : predicted bulk speed [km s⁻¹]
                - 'Name'  (str)       : model display name (single value repeated)
            All series must be on the same time grid as `observed_data`.

    Returns:
        dash_table.DataTable
            A Dash DataTable with one row per statistic and columns for
            "Observed" and each model. Numeric values are formatted to two
            decimals and right-aligned, using global `my_tablelayout`.
    """

    # Prepare column order: Observed first, then each model
    observed_values = observed_data['Obs']
    model_names = [pred_df['Name'][0] for pred_df in predicted_data_list]
    model_series = [pred_df['Model'] for pred_df in predicted_data_list]
    column_order = ['Observed'] + model_names

    # Compute stats
    stats_map = {
        'Observed': descriptiveStats(observed_values)
    }
    for name, series in zip(model_names, model_series):
        stats_map[name] = descriptiveStats(series)

    stat_rows = [
        ('mean',      'Mean'),
        ('median',    'Median'),
        ('std',       'SD'),
        ('min',       'Minimum'),
        ('max',       'Maximum'),
        ('var',       'Variance'),
        ('skewness',  'Skewness'),
        ('kurtosis',  'Kurtosis')
    ]

    # Build DataTable rows
    data = []
    for key, label in stat_rows:
        row = {'Statistic': label}
        for col in column_order:
            row[col] = stats_map[col][key]
        data.append(row)

    # Define columns with formatting
    columns = [{'name': 'Statistic', 'id': 'Statistic'}] + [
        {'name': col, 'id': col, 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)}
        for col in column_order
    ]

    # Align numeric columns right
    style_cell_conditional = [
        {'if': {'column_id': col}, 'textAlign': 'right'} for col in column_order
    ]

    # Create Dash DataTable
    statistics_table = dash_table.DataTable(
        data=data,
        columns=columns,
        style_cell_conditional=style_cell_conditional,
        **my_tablelayout
    )

    return statistics_table


def create_mm_peak_bias_scatter(predicted_data_list, observed_data, threshold_value=450, maximum_dt=2):
    """
    Compute and visualize the relationship between peak prediction bias
    and prediction accuracy for multiple models.

    Parameters:
        predicted_data_list : list[pandas.DataFrame]
            List of model prediction DataFrames, each containing:
                - 'Time'  (datetime64): timestamps
                - 'Model' (float)     : predicted bulk speed [km s⁻¹]
                - 'Name'  (str)       : model display name (single value repeated)
            Timestamps should be aligned with `observed_data`.

        observed_data : pandas.DataFrame
            Observation DataFrame containing:
                - 'Time' (datetime64): timestamps
                - 'Obs'  (float)     : observed bulk speed [km s⁻¹]
                - 'Name' (str)       : label for observations (single value repeated)

        threshold_value : float, optional
            Bulk speed threshold [km s⁻¹] used by `compute_peaks` to detect
            significant events (default = 450).

        maximum_dt : float, optional
            Maximum allowed time difference (in days) for matching predicted
            and observed peaks (default = 2 days).

    Returns:
        plotly.graph_objects.Figure
            Scatter plot with:
                - One point per model, labeled with model name
                - X-axis = peak bias (Predicted ÷ Observed)
                - Y-axis = hit rate (%)
                - Quadrant annotations for interpretation
    """

    observed_data['Time'] = pd.to_datetime(observed_data['Time'])
    model_names = [df['Name'][0] for df in predicted_data_list]

    bias_list = []
    hit_percentages = []
    labels = []

    for df in predicted_data_list:
        model_name = df['Name'][0]
        df['Time'] = pd.to_datetime(df['Time'])

        # Merge predictions with observations
        merged = pd.merge(df, observed_data, on='Time', how='inner')
        if merged.empty:
            continue

        # Compute peaks and associations
        peaks_obs, peaks_pred, hits, misses, false_alarms, time_diffs, data_diffs = compute_peaks(
            merged['Time'], merged['Obs'], merged['Model'], threshold_value, maximum_dt
        )

        num_predicted = len(peaks_pred)
        num_observed = len(peaks_obs)

        if num_predicted == 0 or num_observed == 0:
            continue

        bias = num_predicted / num_observed
        hit_rate = (len(hits) / num_predicted) * 100

        bias_list.append(bias)
        hit_percentages.append(hit_rate)
        labels.append(model_name)

    # Generate distinct colors
    color_map = px.colors.qualitative.Plotly
    point_colors = [color_map[i % len(color_map)] for i in range(len(labels))]

    # Create scatter plot with colored points
    fig = go.Figure()

    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=[bias_list[i]],
            y=[hit_percentages[i]],
            mode='markers+text',
            text=[label],
            textposition='bottom center',
            marker=dict(size=10, color=point_colors[i]),
            name=label,
            showlegend=False,
            hovertemplate=f"Model: {label}<br>Bias: {bias_list[i]:.2f}<br>Hit Rate: {hit_percentages[i]:.1f}%<extra></extra>"
        ))

    # Compute axis limits
    x_min = 0
    x_max = 2
    y_min = 0
    y_max = 101

    # Annotations for quadrants
    annotations = [
        dict(x=x_min + 0.01 * (x_max - x_min), y=y_max - 5, text="Few peaks<br>Well-targeted", showarrow=False, xanchor='left', font=dict(size=11, color='grey')),
        dict(x=x_max - 0.01 * (x_max - x_min), y=y_max - 5, text="Too many peaks<br>Many matches", showarrow=False, xanchor='right', font=dict(size=11, color='grey')),
        dict(x=x_min + 0.01 * (x_max - x_min), y=y_min + 5, text="Few peaks<br>Mostly wrong", showarrow=False, xanchor='left', font=dict(size=11, color='grey')),
        dict(x=x_max - 0.01 * (x_max - x_min), y=y_min + 5, text="Too many peaks<br>Mostly wrong", showarrow=False, xanchor='right', font=dict(size=11, color='grey'))
    ]

    # Add annotations
    for ann in annotations:
        fig.add_annotation(ann)

    fig.update_layout(
        my_figlayout,
        title="Peak Prediction Coverage and Accuracy",
        xaxis_title="Bias (Predicted ÷ Observed)",
        yaxis_title="Success Rate of Predicted Peaks [%]",
        xaxis=dict(showgrid=True, gridcolor='lightgrey', range=[0, max(2, max(bias_list, default=1.5))]),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', range=[0, y_max]),
        height=700
    )

    return fig


def create_mm_overall_error_vs_detection_scatter(predicted_data_list, observed_data, threshold_value=450, maximum_dt=2):
    """
    Compute and visualize the trade-off between overall time series accuracy (RMSE)
    and event detection sensitivity (POD) for multiple models.

    Parameters:
        predicted_data_list : list[pandas.DataFrame]
            List of model prediction DataFrames, each containing:
                - 'Time'  (datetime64): timestamps
                - 'Model' (float)     : predicted bulk speed [km s⁻¹]
                - 'Name'  (str)       : model display name (single value repeated)
            Timestamps should be aligned with `observed_data`.

        observed_data : pandas.DataFrame
            Observation DataFrame containing:
                - 'Time' (datetime64): timestamps
                - 'Obs'  (float)     : observed bulk speed [km s⁻¹]
                - 'Name' (str)       : label for observations (single value repeated)

        threshold_value : float, optional
            Bulk speed threshold [km s⁻¹] used by `compute_peaks` to identify events.
            Default = 450.

        maximum_dt : float, optional
            Maximum allowed time difference (in days) for matching predicted and observed peaks.
            Default = 2 days.

    Returns:
        plotly.graph_objects.Figure
            Scatter plot with:
                - One point per model, labeled with model name
                - X-axis = RMSE [km s⁻¹]
                - Y-axis = POD [%]
                - Quadrant annotations for interpretation
    """

    observed_data['Time'] = pd.to_datetime(observed_data['Time'])
    model_names = [df['Name'][0] for df in predicted_data_list]

    rmse_list = []
    pod_list = []
    labels = []

    for df in predicted_data_list:
        model_name = df['Name'][0]
        df['Time'] = pd.to_datetime(df['Time'])

        merged = pd.merge(df, observed_data, on='Time', how='inner')
        if merged.empty:
            continue

        obs = merged['Obs']
        pred = merged['Model']

        # Compute RMSE
        rmse = np.sqrt(((pred - obs) ** 2).mean())

        # Compute peak associations
        peaks_obs, peaks_pred, hits, misses, false_alarms, time_diffs, data_diffs = compute_peaks(
            merged['Time'], merged['Obs'], merged['Model'], threshold_value, maximum_dt
        )

        TP = len(hits)
        FN = len(misses)

        if TP + FN == 0:
            continue

        pod = 100 * (TP / (TP + FN))

        rmse_list.append(rmse)
        pod_list.append(pod)
        labels.append(model_name)

    # Generate distinct colors
    color_map = px.colors.qualitative.Plotly
    point_colors = [color_map[i % len(color_map)] for i in range(len(labels))]

    # Axis ranges
    x_min = 0
    x_max = max(rmse_list, default=50) * 1.2
    y_min = 0
    y_max = 101

    # Scatter plot
    fig = go.Figure()

    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=[rmse_list[i]],
            y=[pod_list[i]],
            mode='markers+text',
            text=[label],
            textposition='bottom center',
            marker=dict(size=10, color=point_colors[i]),
            name=label,
            showlegend=False,
            hovertemplate=f"Model: {label}<br>RMSE: {rmse_list[i]:.1f} km/s<br>POD: {pod_list[i]:.2f}<extra></extra>"
        ))

    # Annotations for interpretation
    annotations = [
        dict(x=x_min + 0.02 * (x_max - x_min), y=y_max - 5, text="Accurate<br>Sensitive", showarrow=False, xanchor='left', font=dict(size=11, color='grey')),
        dict(x=x_max - 0.02 * (x_max - x_min), y=y_max - 5, text="Sensitive<br>Inaccurate", showarrow=False, xanchor='right', font=dict(size=11, color='grey')),
        dict(x=x_min + 0.02 * (x_max - x_min), y=y_min + 5, text="Accurate<br>Insensitive", showarrow=False, xanchor='left', font=dict(size=11, color='grey')),
        dict(x=x_max - 0.02 * (x_max - x_min), y=y_min + 5, text="Inaccurate<br>Insensitive", showarrow=False, xanchor='right', font=dict(size=11, color='grey'))
    ]

    for ann in annotations:
        fig.add_annotation(ann)

    fig.update_layout(
        my_figlayout,
        title="Time Series Accuracy and Event Sensitivity",
        xaxis_title="RMSE [km s⁻¹]",
        yaxis_title="Correctly Predicted Observed Peaks [%]",
        xaxis=dict(showgrid=True, gridcolor='lightgrey', range=[x_min, x_max]),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', range=[y_min, y_max]),
        height=700
    )

    return fig