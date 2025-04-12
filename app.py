import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from data_processor import process_data, get_data_summary, export_data
from visualizer import generate_visualization, get_visualization_options
from analyzer import perform_statistical_analysis, detect_trends
import subprocess
import sys
import os

# Configure the Streamlit page
st.set_page_config(
    page_title="DevInsight - Data Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'api_running' not in st.session_state:
    st.session_state.api_running = False

def start_api_server():
    """Start the Flask API server"""
    if not st.session_state.api_running:
        try:
            # Start the Flask API in a separate process
            # Note: In production, this would be handled differently
            subprocess.Popen([sys.executable, 'api.py'])
            st.session_state.api_running = True
            return True
        except Exception as e:
            st.error(f"Failed to start API server: {e}")
            return False
    return True

# Main title
st.title("DevInsight - Data Analysis Tool")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Import", "Data Exploration", "Visualization", "Analysis", "API Integration", "Export"])

# Data Import Page
if page == "Data Import":
    st.header("Import Your Data")
    
    st.write("""
    Upload your data file to get started. We support CSV, Excel, and JSON formats.
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])
    
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                data = pd.read_csv(uploaded_file)
                st.session_state.filename = uploaded_file.name
            
            elif file_extension == 'xlsx':
                data = pd.read_excel(uploaded_file)
                st.session_state.filename = uploaded_file.name
            
            elif file_extension == 'json':
                content = uploaded_file.getvalue().decode('utf-8')
                data = pd.DataFrame(json.loads(content))
                st.session_state.filename = uploaded_file.name
            
            # Process the data
            st.session_state.data = process_data(data)
            
            # Display success message and data preview
            st.success(f"Successfully loaded {uploaded_file.name}")
            st.subheader("Data Preview")
            st.dataframe(st.session_state.data.head(10))
            
            # Display basic statistics
            st.subheader("Basic Data Information")
            st.write(f"Total rows: {len(st.session_state.data)}")
            st.write(f"Total columns: {len(st.session_state.data.columns)}")
            
            # Column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Data Type': st.session_state.data.dtypes,
                'Non-Null Count': st.session_state.data.notnull().sum(),
                'Null Count': st.session_state.data.isnull().sum()
            })
            st.dataframe(col_info)
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Option to load sample data
    st.subheader("API Import")
    with st.expander("Import data from an API"):
        api_url = st.text_input("API URL")
        auth_token = st.text_input("Authorization Token (if required)", type="password")
        
        if st.button("Import from API"):
            try:
                import requests
                headers = {}
                if auth_token:
                    headers['Authorization'] = f"Bearer {auth_token}"
                
                response = requests.get(api_url, headers=headers)
                
                if response.status_code == 200:
                    try:
                        json_data = response.json()
                        data = pd.json_normalize(json_data)
                        st.session_state.data = process_data(data)
                        st.session_state.filename = "api_data.json"
                        
                        st.success("Successfully loaded data from API")
                        st.subheader("Data Preview")
                        st.dataframe(st.session_state.data.head(10))
                    except Exception as e:
                        st.error(f"Error processing API response: {e}")
                else:
                    st.error(f"API request failed with status code: {response.status_code}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")

# Data Exploration Page
elif page == "Data Exploration":
    st.header("Explore Your Data")
    
    if st.session_state.data is not None:
        st.subheader(f"Current dataset: {st.session_state.filename}")
        
        # Column selection for data filtering
        all_columns = st.session_state.data.columns.tolist()
        
        # Data filtering options
        st.subheader("Filter Data")
        with st.form("filter_form"):
            filter_column = st.selectbox("Select column to filter", all_columns)
            
            # Determine filter type based on data type
            column_type = st.session_state.data[filter_column].dtype
            
            if np.issubdtype(column_type, np.number):
                min_val = float(st.session_state.data[filter_column].min())
                max_val = float(st.session_state.data[filter_column].max())
                filter_range = st.slider(f"Range for {filter_column}", min_val, max_val, (min_val, max_val))
                
                submitted = st.form_submit_button("Apply Filter")
                if submitted:
                    filtered_data = st.session_state.data[(st.session_state.data[filter_column] >= filter_range[0]) & 
                                                         (st.session_state.data[filter_column] <= filter_range[1])]
                    st.write(f"Filtered data contains {len(filtered_data)} rows")
                    st.dataframe(filtered_data)
            else:
                unique_values = st.session_state.data[filter_column].unique().tolist()
                selected_values = st.multiselect(f"Select values for {filter_column}", unique_values, unique_values[:5] if len(unique_values) > 5 else unique_values)
                
                submitted = st.form_submit_button("Apply Filter")
                if submitted:
                    filtered_data = st.session_state.data[st.session_state.data[filter_column].isin(selected_values)]
                    st.write(f"Filtered data contains {len(filtered_data)} rows")
                    st.dataframe(filtered_data)
        
        # Data querying
        st.subheader("Query Data")
        with st.expander("Run a query on your data"):
            st.write("You can use Python syntax to query your data. Use 'df' to refer to your dataset.")
            query = st.text_area("Enter your query:", "df[df['column_name'] > value]")
            
            if st.button("Run Query"):
                try:
                    df = st.session_state.data  # Create a reference to the data for the query
                    result = eval(query)
                    if isinstance(result, pd.DataFrame):
                        st.write(f"Query returned {len(result)} rows")
                        st.dataframe(result)
                    else:
                        st.write("Result:")
                        st.write(result)
                except Exception as e:
                    st.error(f"Error executing query: {e}")
        
        # Data summary statistics
        st.subheader("Data Summary")
        summary_options = st.multiselect(
            "Select columns for summary statistics",
            all_columns,
            all_columns[:5] if len(all_columns) > 5 else all_columns
        )
        
        if summary_options:
            summary = get_data_summary(st.session_state.data, summary_options)
            st.dataframe(summary)
        
        # Correlation heatmap for numerical columns
        st.subheader("Correlation Analysis")
        numeric_columns = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) > 1:
            corr_columns = st.multiselect(
                "Select columns for correlation analysis",
                numeric_columns,
                numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns
            )
            
            if corr_columns and len(corr_columns) > 1:
                corr_fig = generate_visualization(
                    st.session_state.data,
                    viz_type="correlation",
                    x=corr_columns,
                    y=None
                )
                st.plotly_chart(corr_fig, use_container_width=True)
            elif corr_columns:
                st.info("Please select at least 2 columns for correlation analysis")
        else:
            st.info("Not enough numerical columns for correlation analysis")
    else:
        st.info("Please import data first on the Data Import page")

# Visualization Page
elif page == "Visualization":
    st.header("Visualize Your Data")
    
    if st.session_state.data is not None:
        # Get columns by data type
        numeric_columns = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist()
        date_columns = [col for col in st.session_state.data.columns if st.session_state.data[col].dtype == 'datetime64[ns]']
        
        # Visualization options
        viz_options = get_visualization_options(st.session_state.data)
        
        st.subheader("Create Visualization")
        
        viz_type = st.selectbox("Select visualization type", list(viz_options.keys()))
        
        # Dynamic form based on the visualization type
        with st.form("viz_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                if viz_options[viz_type]['x_axis'] == 'numeric':
                    x = st.selectbox("Select X-axis", numeric_columns)
                elif viz_options[viz_type]['x_axis'] == 'categorical':
                    x = st.selectbox("Select X-axis", categorical_columns if categorical_columns else numeric_columns)
                elif viz_options[viz_type]['x_axis'] == 'any':
                    x = st.selectbox("Select X-axis", st.session_state.data.columns.tolist())
                elif viz_options[viz_type]['x_axis'] == 'multiple':
                    x = st.multiselect("Select columns", numeric_columns, numeric_columns[:3] if len(numeric_columns) > 3 else numeric_columns)
                else:
                    x = None
            
            with col2:
                if viz_options[viz_type]['y_axis'] == 'numeric':
                    y = st.selectbox("Select Y-axis", numeric_columns)
                elif viz_options[viz_type]['y_axis'] == 'categorical':
                    y = st.selectbox("Select Y-axis", categorical_columns if categorical_columns else numeric_columns)
                elif viz_options[viz_type]['y_axis'] == 'any':
                    y = st.selectbox("Select Y-axis", st.session_state.data.columns.tolist())
                elif viz_options[viz_type]['y_axis'] == 'multiple':
                    y = st.multiselect("Select columns", numeric_columns, numeric_columns[:3] if len(numeric_columns) > 3 else numeric_columns)
                else:
                    y = None
            
            # Color option for applicable charts
            if viz_options[viz_type].get('color', False):
                color = st.selectbox("Color by", [None] + categorical_columns)
            else:
                color = None
            
            # Additional options based on chart type
            if viz_type == 'histogram':
                bins = st.slider("Number of bins", 5, 100, 20)
            else:
                bins = 20
            
            title = st.text_input("Chart Title", f"{viz_type.capitalize()} of {x if isinstance(x, str) else 'selected data'}")
            
            submitted = st.form_submit_button("Generate Visualization")
        
        if submitted:
            try:
                fig = generate_visualization(
                    st.session_state.data,
                    viz_type=viz_type,
                    x=x,
                    y=y,
                    color=color,
                    bins=bins,
                    title=title
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to save the visualization
                buffer = io.BytesIO()
                fig.write_html(buffer)
                st.download_button(
                    label="Download Visualization (HTML)",
                    data=buffer.getvalue(),
                    file_name=f"{viz_type}_{x}_{y if y is not None else ''}.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"Error generating visualization: {e}")
        
        # Dashboard creator
        st.subheader("Create Simple Dashboard")
        with st.expander("Dashboard Creator"):
            st.write("Select multiple visualizations to create a dashboard")
            
            dash_cols = st.number_input("Number of columns in dashboard", min_value=1, max_value=3, value=2)
            
            viz_containers = []
            for i in range(3):  # Allow up to 3 visualizations in the dashboard
                with st.container():
                    st.write(f"Visualization {i+1}")
                    viz_type = st.selectbox(f"Type {i+1}", list(viz_options.keys()), key=f"dash_viz_type_{i}")
                    
                    if viz_options[viz_type]['x_axis'] == 'numeric':
                        x = st.selectbox(f"X-axis {i+1}", numeric_columns, key=f"dash_x_{i}")
                    elif viz_options[viz_type]['x_axis'] == 'categorical':
                        x = st.selectbox(f"X-axis {i+1}", categorical_columns if categorical_columns else numeric_columns, key=f"dash_x_{i}")
                    elif viz_options[viz_type]['x_axis'] == 'any':
                        x = st.selectbox(f"X-axis {i+1}", st.session_state.data.columns.tolist(), key=f"dash_x_{i}")
                    elif viz_options[viz_type]['x_axis'] == 'multiple':
                        x = st.multiselect(f"Columns {i+1}", numeric_columns, numeric_columns[:2], key=f"dash_x_{i}")
                    else:
                        x = None
                    
                    if viz_options[viz_type]['y_axis'] == 'numeric':
                        y = st.selectbox(f"Y-axis {i+1}", numeric_columns, key=f"dash_y_{i}")
                    elif viz_options[viz_type]['y_axis'] == 'categorical':
                        y = st.selectbox(f"Y-axis {i+1}", categorical_columns if categorical_columns else numeric_columns, key=f"dash_y_{i}")
                    elif viz_options[viz_type]['y_axis'] == 'any':
                        y = st.selectbox(f"Y-axis {i+1}", st.session_state.data.columns.tolist(), key=f"dash_y_{i}")
                    elif viz_options[viz_type]['y_axis'] == 'multiple':
                        y = st.multiselect(f"Columns {i+1}", numeric_columns, numeric_columns[:2], key=f"dash_y_{i}")
                    else:
                        y = None
                    
                    viz_containers.append((viz_type, x, y))
                    
                    if i < 2:  # No separator after the last item
                        st.markdown("---")
            
            if st.button("Generate Dashboard"):
                # Create the dashboard with the specified columns
                dashboard_cols = st.columns(dash_cols)
                for i, (viz_type, x, y) in enumerate(viz_containers):
                    try:
                        fig = generate_visualization(
                            st.session_state.data,
                            viz_type=viz_type,
                            x=x,
                            y=y
                        )
                        col_idx = i % dash_cols
                        with dashboard_cols[col_idx]:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        with dashboard_cols[i % dash_cols]:
                            st.error(f"Error in visualization {i+1}: {e}")
    else:
        st.info("Please import data first on the Data Import page")

# Analysis Page
elif page == "Analysis":
    st.header("Analyze Your Data")
    
    if st.session_state.data is not None:
        # Get columns by data type
        numeric_columns = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Statistical Analysis Section
        st.subheader("Statistical Analysis")
        
        analysis_type = st.selectbox(
            "Select analysis type",
            ["Descriptive Statistics", "Correlation Analysis", "Regression Analysis", "Time Series Analysis", "Trend Detection"]
        )
        
        if analysis_type == "Descriptive Statistics":
            selected_columns = st.multiselect(
                "Select columns for descriptive statistics",
                st.session_state.data.columns.tolist(),
                numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns
            )
            
            if selected_columns:
                stats_df = perform_statistical_analysis(
                    st.session_state.data,
                    analysis_type="descriptive",
                    columns=selected_columns
                )
                st.dataframe(stats_df)
                
                # Visualize the distribution of selected columns
                if st.checkbox("Visualize distributions"):
                    for col in selected_columns:
                        if pd.api.types.is_numeric_dtype(st.session_state.data[col]):
                            fig = generate_visualization(
                                st.session_state.data,
                                viz_type="histogram",
                                x=col,
                                y=None,
                                title=f"Distribution of {col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Correlation Analysis":
            if len(numeric_columns) > 1:
                corr_columns = st.multiselect(
                    "Select columns for correlation analysis",
                    numeric_columns,
                    numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns
                )
                
                if corr_columns and len(corr_columns) > 1:
                    corr_method = st.radio("Correlation method", ["pearson", "spearman", "kendall"])
                    
                    corr_matrix = perform_statistical_analysis(
                        st.session_state.data,
                        analysis_type="correlation",
                        columns=corr_columns,
                        method=corr_method
                    )
                    
                    st.write("Correlation Matrix:")
                    st.dataframe(corr_matrix)
                    
                    # Visualize the correlation matrix
                    fig = generate_visualization(
                        st.session_state.data,
                        viz_type="correlation",
                        x=corr_columns,
                        y=None,
                        title=f"{corr_method.capitalize()} Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Highlight strong correlations
                    strong_corr = []
                    for i in range(len(corr_columns)):
                        for j in range(i+1, len(corr_columns)):
                            if abs(corr_matrix.iloc[i, j]) > 0.7:
                                strong_corr.append((corr_columns[i], corr_columns[j], corr_matrix.iloc[i, j]))
                    
                    if strong_corr:
                        st.subheader("Strong Correlations")
                        for col1, col2, corr_val in strong_corr:
                            st.write(f"{col1} and {col2}: {corr_val:.4f}")
                            fig = generate_visualization(
                                st.session_state.data,
                                viz_type="scatter",
                                x=col1,
                                y=col2,
                                title=f"Scatter Plot: {col1} vs {col2}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least 2 columns for correlation analysis")
            else:
                st.info("Not enough numerical columns for correlation analysis")
        
        elif analysis_type == "Regression Analysis":
            if len(numeric_columns) > 1:
                st.write("Simple Linear Regression Analysis")
                
                x_col = st.selectbox("Select predictor (X) variable", numeric_columns)
                y_col = st.selectbox("Select target (Y) variable", [col for col in numeric_columns if col != x_col])
                
                if st.button("Run Regression Analysis"):
                    reg_results = perform_statistical_analysis(
                        st.session_state.data,
                        analysis_type="regression",
                        x=x_col,
                        y=y_col
                    )
                    
                    st.write("Regression Results:")
                    st.write(reg_results['summary'])
                    
                    st.write(f"R-squared: {reg_results['r_squared']:.4f}")
                    st.write(f"Adjusted R-squared: {reg_results['adj_r_squared']:.4f}")
                    st.write(f"F-statistic: {reg_results['f_statistic']:.4f}")
                    st.write(f"P-value: {reg_results['p_value']:.4f}")
                    
                    # Visualize the regression line
                    fig = reg_results['figure']
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numerical columns for regression analysis")
        
        elif analysis_type == "Time Series Analysis":
            date_columns = [col for col in st.session_state.data.columns if 
                           ('date' in col.lower() or 'time' in col.lower() or 
                            pd.api.types.is_datetime64_any_dtype(st.session_state.data[col]))]
            
            if date_columns:
                date_col = st.selectbox("Select date/time column", date_columns)
                
                # Try to convert the selected column to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(st.session_state.data[date_col]):
                    try:
                        st.session_state.data[date_col] = pd.to_datetime(st.session_state.data[date_col])
                        st.success(f"Converted {date_col} to datetime format")
                    except Exception as e:
                        st.error(f"Could not convert {date_col} to datetime: {e}")
                        st.info("Please select a valid date/time column")
                        date_col = None
                
                if date_col:
                    value_col = st.selectbox("Select value column to analyze", numeric_columns)
                    
                    if st.button("Run Time Series Analysis"):
                        ts_results = perform_statistical_analysis(
                            st.session_state.data,
                            analysis_type="time_series",
                            date_column=date_col,
                            value_column=value_col
                        )
                        
                        st.write("Time Series Analysis Results:")
                        
                        # Original time series
                        st.plotly_chart(ts_results['original_series'], use_container_width=True)
                        
                        # Decomposition
                        if 'decomposition' in ts_results:
                            st.subheader("Time Series Decomposition")
                            st.plotly_chart(ts_results['decomposition'], use_container_width=True)
                            
                            if 'seasonality_test' in ts_results:
                                st.write(f"Seasonality Test p-value: {ts_results['seasonality_test']:.4f}")
                                if ts_results['seasonality_test'] < 0.05:
                                    st.write("The time series shows significant seasonality")
                                else:
                                    st.write("No significant seasonality detected")
                        
                        # Autocorrelation
                        if 'autocorrelation' in ts_results:
                            st.subheader("Autocorrelation Analysis")
                            st.plotly_chart(ts_results['autocorrelation'], use_container_width=True)
                            
                            if ts_results.get('stationarity_test'):
                                st.write(f"Stationarity Test p-value: {ts_results['stationarity_test']:.4f}")
                                if ts_results['stationarity_test'] < 0.05:
                                    st.write("The time series is stationary")
                                else:
                                    st.write("The time series is not stationary")
            else:
                st.info("No date/time columns found in the dataset. Please ensure your time column is properly formatted.")
        
        elif analysis_type == "Trend Detection":
            if numeric_columns:
                target_col = st.selectbox("Select column for trend detection", numeric_columns)
                
                if st.button("Detect Trends"):
                    trend_results = detect_trends(
                        st.session_state.data,
                        target_column=target_col
                    )
                    
                    st.write("Trend Detection Results:")
                    
                    # Mann-Kendall test
                    st.write(f"Mann-Kendall Trend Test:")
                    st.write(f"Trend: {'Increasing' if trend_results['mann_kendall']['trend'] > 0 else 'Decreasing' if trend_results['mann_kendall']['trend'] < 0 else 'No trend'}")
                    st.write(f"p-value: {trend_results['mann_kendall']['p_value']:.4f}")
                    st.write(f"Tau: {trend_results['mann_kendall']['tau']:.4f}")
                    
                    # Visualize the trend
                    st.plotly_chart(trend_results['trend_plot'], use_container_width=True)
                    
                    # Outlier detection
                    if 'outliers' in trend_results:
                        st.subheader("Outlier Detection")
                        outlier_count = trend_results['outliers'].sum()
                        st.write(f"Detected {outlier_count} outliers in the data")
                        
                        if outlier_count > 0:
                            st.plotly_chart(trend_results['outlier_plot'], use_container_width=True)
            else:
                st.info("No numerical columns found for trend detection")
    else:
        st.info("Please import data first on the Data Import page")

# API Integration Page
elif page == "API Integration":
    st.header("API Integration")
    
    st.write("""
    This page provides information on how to integrate with the DevInsight API.
    You can use these endpoints to programmatically access and analyze your data.
    """)
    
    # Start the API server if it's not already running
    api_status = start_api_server()
    
    if api_status:
        st.success("API server is running on port 8000")
        
        st.subheader("API Documentation")
        
        st.markdown("""
        ### Authentication
        No authentication is required for local development.
        
        ### Base URL
        ```
        http://localhost:8000
        ```
        
        ### Endpoints
        
        #### 1. Get Data
        
        ```http
        GET /api/data
        ```
        
        Returns the current dataset as JSON.
        
        #### 2. Get Data Summary
        
        ```http
        GET /api/summary
        ```
        
        Returns summary statistics for the current dataset.
        
        #### 3. Generate Visualization
        
        ```http
        POST /api/visualize
        ```
        
        Generate a visualization and return it as a base64-encoded image.
        
        **Request Body:**
        ```json
        {
            "viz_type": "bar",
            "x": "column_name",
            "y": "column_name",
            "title": "My Visualization"
        }
        ```
        
        #### 4. Perform Analysis
        
        ```http
        POST /api/analyze
        ```
        
        Perform statistical analysis on the data.
        
        **Request Body:**
        ```json
        {
            "analysis_type": "correlation",
            "columns": ["column1", "column2", "column3"],
            "method": "pearson"
        }
        ```
        """)
        