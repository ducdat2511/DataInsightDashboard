import pandas as pd
import numpy as np
import io
import json

def process_data(data):
    """
    Process the imported data to make it suitable for analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to process
        
    Returns:
    --------
    pandas.DataFrame
        Processed data
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Convert date columns to proper datetime format
    for col in df.columns:
        # Check if the column might be a date
        if col.lower().find('date') >= 0 or col.lower().find('time') >= 0:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                # If conversion fails, leave as is
                pass
    
    # Handle basic data cleaning
    # 1. Drop completely empty columns
    df = df.dropna(axis=1, how='all')
    
    # 2. For numeric columns, replace NaN with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # 3. For categorical columns, replace NaN with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if len(df[col].dropna()) > 0:  # Make sure there are non-NA values
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def get_data_summary(data, columns=None):
    """
    Generate summary statistics for the data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to summarize
    columns : list, optional
        Specific columns to summarize. If None, all columns are used.
        
    Returns:
    --------
    pandas.DataFrame
        Summary statistics
    """
    if columns is None:
        columns = data.columns
    
    # Get only the specified columns
    df = data[columns].copy()
    
    # Split into numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # Summary for numeric columns
    summary = pd.DataFrame()
    
    if len(numeric_cols) > 0:
        num_summary = df[numeric_cols].describe().T
        num_summary['median'] = df[numeric_cols].median()
        num_summary['missing'] = df[numeric_cols].isnull().sum()
        num_summary['missing_pct'] = df[numeric_cols].isnull().sum() / len(df) * 100
        
        summary = pd.concat([summary, num_summary])
    
    # Summary for categorical columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            # Get value counts for the column
            value_counts = df[col].value_counts()
            
            # Create a temporary dataframe for this column
            temp_df = pd.DataFrame({
                'count': [len(df[col].dropna())],
                'unique': [df[col].nunique()],
                'top': [value_counts.index[0] if len(value_counts) > 0 else None],
                'freq': [value_counts.iloc[0] if len(value_counts) > 0 else 0],
                'missing': [df[col].isnull().sum()],
                'missing_pct': [df[col].isnull().sum() / len(df) * 100]
            }, index=[col])
            
            summary = pd.concat([summary, temp_df])
    
    return summary

def export_data(data, format='csv'):
    """
    Export the data to the specified format.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to export
    format : str, optional
        The format to export to ('csv', 'excel', or 'json')
        
    Returns:
    --------
    bytes
        The exported data as bytes
    """
    buffer = io.BytesIO()
    
    if format == 'csv':
        data.to_csv(buffer, index=False)
    elif format == 'excel':
        data.to_excel(buffer, index=False)
    elif format == 'json':
        buffer.write(data.to_json(orient='records').encode('utf-8'))
    else:
        raise ValueError(f"Unsupported export format: {format}")
    
    buffer.seek(0)
    return buffer.getvalue()
