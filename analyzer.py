import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def perform_statistical_analysis(data, analysis_type, columns=None, x=None, y=None, date_column=None, value_column=None, method="pearson"):
    """
    Perform statistical analysis on the data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to analyze
    analysis_type : str
        The type of analysis to perform ('descriptive', 'correlation', 'regression', 'time_series')
    columns : list, optional
        The columns to include in the analysis (for 'descriptive' and 'correlation')
    x : str, optional
        The predictor variable (for 'regression')
    y : str, optional
        The target variable (for 'regression')
    date_column : str, optional
        The date column (for 'time_series')
    value_column : str, optional
        The value column (for 'time_series')
    method : str, optional
        The correlation method to use ('pearson', 'spearman', or 'kendall')
        
    Returns:
    --------
    various
        The results of the analysis, type depends on analysis_type
    """
    if analysis_type == "descriptive":
        # Basic descriptive statistics
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        return data[columns].describe().T
    
    elif analysis_type == "correlation":
        # Correlation analysis
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        return data[columns].corr(method=method)
    
    elif analysis_type == "regression":
        # Simple linear regression
        X = sm.add_constant(data[x])
        model = sm.OLS(data[y], X).fit()
        
        # Create a scatter plot with regression line
        fig = px.scatter(data, x=x, y=y, title=f"Linear Regression: {y} vs {x}")
        
        # Add regression line
        x_range = np.linspace(data[x].min(), data[x].max(), 100)
        y_pred = model.params[0] + model.params[1] * x_range
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                mode="lines",
                name="Regression Line",
                line=dict(color="red", width=2)
            )
        )
        
        # Add formula annotation
        formula = f"{y} = {model.params[0]:.4f} + {model.params[1]:.4f} × {x}"
        fig.add_annotation(
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            text=formula,
            showarrow=False,
            font=dict(size=14),
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )
        
        return {
            "summary": model.summary().as_text(),
            "params": model.params.to_dict(),
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "f_statistic": model.fvalue,
            "p_value": model.f_pvalue,
            "figure": fig
        }
    
    elif analysis_type == "time_series":
        # Time series analysis
        # Ensure the date column is the index
        ts_data = data.copy()
        if date_column is not None:
            ts_data = ts_data.set_index(date_column)
        
        # Sort the index (important for time series)
        ts_data = ts_data.sort_index()
        
        # Select only the value column
        if value_column is not None:
            ts_series = ts_data[value_column]
        else:
            # Use the first numeric column
            numeric_cols = ts_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ts_series = ts_data[numeric_cols[0]]
            else:
                raise ValueError("No numeric columns found for time series analysis")
        
        # Create a time series plot
        fig_ts = px.line(
            ts_series,
            x=ts_series.index,
            y=ts_series.values,
            title=f"Time Series: {value_column or 'Value'} over time"
        )
        
        results = {
            "original_series": fig_ts
        }
        
        # Try to decompose the time series if it has enough data points
        try:
            if len(ts_series) >= 14:  # Minimum required for seasonal_decompose with default settings
                # Apply seasonal decomposition
                decomposition = seasonal_decompose(ts_series, model='additive', period=min(7, len(ts_series) // 2))
                
                # Create subplots for decomposition
                fig_decomp = make_subplots(
                    rows=4,
                    cols=1,
                    subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
                    shared_xaxes=True,
                    vertical_spacing=0.05
                )
                
                # Original series
                fig_decomp.add_trace(
                    go.Scatter(x=ts_series.index, y=ts_series.values, mode="lines", name="Observed"),
                    row=1, col=1
                )
                
                # Trend component
                fig_decomp.add_trace(
                    go.Scatter(x=ts_series.index, y=decomposition.trend, mode="lines", name="Trend"),
                    row=2, col=1
                )
                
                # Seasonal component
                fig_decomp.add_trace(
                    go.Scatter(x=ts_series.index, y=decomposition.seasonal, mode="lines", name="Seasonal"),
                    row=3, col=1
                )
                
                # Residual component
                fig_decomp.add_trace(
                    go.Scatter(x=ts_series.index, y=decomposition.resid, mode="lines", name="Residual"),
                    row=4, col=1
                )
                
                # Update layout
                fig_decomp.update_layout(
                    height=800,
                    title_text="Time Series Decomposition",
                    showlegend=False
                )
                
                results["decomposition"] = fig_decomp
                
                # Seasonality test (using Kruskal-Wallis H-test)
                # Group by period and perform the test
                try:
                    period_groups = [ts_series.values[i::7] for i in range(7)]
                    period_groups = [group for group in period_groups if len(group) > 1]
                    if len(period_groups) > 1:
                        _, seasonality_pvalue = stats.kruskal(*period_groups)
                        results["seasonality_test"] = seasonality_pvalue
                except:
                    # Skip if seasonality test fails
                    pass
        except:
            # Skip decomposition if it fails
            pass
        
        # ACF and PACF plots
        try:
            # Create ACF and PACF plots
            fig_acf_pacf = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=["Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"],
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            # Calculate ACF values
            acf_values = acf(ts_series.dropna(), nlags=40)
            lags = range(len(acf_values))
            
            # ACF plot
            fig_acf_pacf.add_trace(
                go.Bar(x=lags, y=acf_values, name="ACF"),
                row=1, col=1
            )
            
            # Add confidence interval for ACF
            ci = 1.96 / np.sqrt(len(ts_series))
            fig_acf_pacf.add_trace(
                go.Scatter(
                    x=lags,
                    y=[ci] * len(lags),
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="95% CI"
                ),
                row=1, col=1
            )
            
            fig_acf_pacf.add_trace(
                go.Scatter(
                    x=lags,
                    y=[-ci] * len(lags),
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Calculate PACF values
            pacf_values = pacf(ts_series.dropna(), nlags=40)
            
            # PACF plot
            fig_acf_pacf.add_trace(
                go.Bar(x=lags, y=pacf_values, name="PACF"),
                row=2, col=1
            )
            
            # Add confidence interval for PACF
            fig_acf_pacf.add_trace(
                go.Scatter(
                    x=lags,
                    y=[ci] * len(lags),
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig_acf_pacf.add_trace(
                go.Scatter(
                    x=lags,
                    y=[-ci] * len(lags),
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Update layout
            fig_acf_pacf.update_layout(
                height=600,
                title_text="ACF and PACF Analysis",
                showlegend=False
            )
            
            results["autocorrelation"] = fig_acf_pacf
            
            # Stationarity test (ADF test)
            try:
                adf_result = adfuller(ts_series.dropna())
                results["stationarity_test"] = adf_result[1]  # p-value
            except:
                # Skip if stationarity test fails
                pass
        except:
            # Skip if ACF/PACF plots fail
            pass
        
        return results
    
    else:
        raise ValueError(f"Unsupported analysis type: {analysis_type}")

def detect_trends(data, target_column, window=5):
    """
    Detect trends and anomalies in the data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to analyze
    target_column : str
        The column to analyze for trends
    window : int, optional
        The window size for moving average
        
    Returns:
    --------
    dict
        Trend detection results
    """
    # Make a copy of the data
    df = data.copy()
    
    # Ensure the target column is numeric
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target column '{target_column}' must be numeric")
    
    # Calculate moving average
    df['rolling_mean'] = df[target_column].rolling(window=window, center=True).mean()
    
    # Perform Mann-Kendall trend test
    try:
        from pymannkendall import original_test
        trend_result = original_test(df[target_column].dropna())
        
        mann_kendall = {
            "trend": 1 if trend_result.trend == "increasing" else (-1 if trend_result.trend == "decreasing" else 0),
            "p_value": trend_result.p,
            "tau": trend_result.Tau,
            "s": trend_result.s,
            "var_s": trend_result.var_s,
            "z": trend_result.z
        }
    except ImportError:
        # If pymannkendall is not available, use a simple correlation with index
        data_clean = df[target_column].dropna()
        corr, p_value = stats.pearsonr(range(len(data_clean)), data_clean)
        
        mann_kendall = {
            "trend": 1 if corr > 0 else (-1 if corr < 0 else 0),
            "p_value": p_value,
            "tau": corr,  # Not exactly Kendall's tau, but a correlation measure
            "s": None,
            "var_s": None,
            "z": None
        }
    
    # Create trend plot
    fig_trend = go.Figure()
    
    # Original data
    fig_trend.add_trace(
        go.Scatter(
            x=list(range(len(df))),
            y=df[target_column],
            mode="markers+lines",
            name=target_column,
            line=dict(color="blue", width=1),
            marker=dict(size=6)
        )
    )
    
    # Rolling mean
    fig_trend.add_trace(
        go.Scatter(
            x=list(range(len(df))),
            y=df['rolling_mean'],
            mode="lines",
            name=f"{window}-point Moving Average",
            line=dict(color="red", width=2)
        )
    )
    
    # Linear trend line
    x_vals = np.array(range(len(df)))
    y_vals = df[target_column].values
    
    # Handle missing values
    mask = ~np.isnan(y_vals)
    x_vals_clean = x_vals[mask]
    y_vals_clean = y_vals[mask]
    
    if len(x_vals_clean) > 1:
        slope, intercept = np.polyfit(x_vals_clean, y_vals_clean, 1)
        trend_line = intercept + slope * x_vals
        
        fig_trend.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=trend_line,
                mode="lines",
                name="Linear Trend",
                line=dict(color="green", width=2, dash="dash")
            )
        )
    
    # Update layout
    trend_direction = "Increasing" if mann_kendall["trend"] > 0 else "Decreasing" if mann_kendall["trend"] < 0 else "No"
    significant = "Significant" if mann_kendall["p_value"] < 0.05 else "Non-significant"
    fig_trend.update_layout(
        title=f"{trend_direction} Trend Detected ({significant}, p={mann_kendall['p_value']:.4f})",
        xaxis_title="Index",
        yaxis_title=target_column,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
    )
    
    # Detect outliers using the Isolation Forest algorithm
    try:
        # Reshape the data for scikit-learn
        X = df[target_column].values.reshape(-1, 1)
        
        # Replace NaN values with column mean
        mean_val = np.nanmean(X)
        X[np.isnan(X)] = mean_val
        
        # Apply isolation forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso_forest.fit_predict(X)
        
        # Convert to boolean (True for outliers, False for inliers)
        outliers = outliers == -1
        
        # Create outlier plot
        fig_outlier = go.Figure()
        
        # Original data
        fig_outlier.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=df[target_column],
                mode="markers+lines",
                name=target_column,
                line=dict(color="blue", width=1),
                marker=dict(
                    size=6,
                    color=["red" if o else "blue" for o in outliers]
                )
            )
        )
        
        # Highlight outliers
        outlier_indices = np.where(outliers)[0]
        outlier_values = [df[target_column].iloc[i] for i in outlier_indices]
        
        fig_outlier.add_trace(
            go.Scatter(
                x=outlier_indices,
                y=outlier_values,
                mode="markers",
                name="Outliers",
                marker=dict(
                    size=10,
                    color="red",
                    symbol="circle-open",
                    line=dict(width=2)
                )
            )
        )
        
        # Update layout
        fig_outlier.update_layout(
            title=f"Outlier Detection for {target_column}",
            xaxis_title="Index",
            yaxis_title=target_column,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
        )
        
        return {
            "mann_kendall": mann_kendall,
            "trend_plot": fig_trend,
            "outliers": pd.Series(outliers, index=df.index),
            "outlier_plot": fig_outlier
        }
    except:
        # Return without outlier detection if it fails
        return {
            "mann_kendall": mann_kendall,
            "trend_plot": fig_trend
        }
