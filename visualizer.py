import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import base64
import io

def get_visualization_options(data):
    """
    Get available visualization options based on the data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to visualize
    
    Returns:
    --------
    dict
        Dictionary of visualization options with their requirements
    """
    # Define visualization options and their axis requirements
    options = {
        "bar": {
            "x_axis": "any",
            "y_axis": "numeric",
            "color": True,
            "description": "Bar chart for categorical comparisons"
        },
        "line": {
            "x_axis": "any",
            "y_axis": "numeric",
            "color": True,
            "description": "Line chart for trends over a continuous variable"
        },
        "scatter": {
            "x_axis": "numeric",
            "y_axis": "numeric",
            "color": True,
            "description": "Scatter plot for relationships between two numeric variables"
        },
        "histogram": {
            "x_axis": "numeric",
            "y_axis": None,
            "color": False,
            "description": "Histogram for distribution of a numeric variable"
        },
        "box": {
            "x_axis": "categorical",
            "y_axis": "numeric",
            "color": True,
            "description": "Box plot for distribution comparison across categories"
        },
        "violin": {
            "x_axis": "categorical",
            "y_axis": "numeric",
            "color": True,
            "description": "Violin plot for detailed distribution comparison"
        },
        "pie": {
            "x_axis": "categorical",
            "y_axis": "numeric",
            "color": False,
            "description": "Pie chart for part-to-whole relationships"
        },
        "heatmap": {
            "x_axis": "categorical",
            "y_axis": "categorical",
            "color": False,
            "description": "Heatmap for visualizing matrix data"
        },
        "correlation": {
            "x_axis": "multiple",
            "y_axis": None,
            "color": False,
            "description": "Correlation matrix for numeric variables"
        },
        "area": {
            "x_axis": "any",
            "y_axis": "numeric",
            "color": True,
            "description": "Area chart for cumulative totals over time"
        },
        "treemap": {
            "x_axis": "categorical",
            "y_axis": "numeric",
            "color": True,
            "description": "Treemap for hierarchical data"
        },
        "sunburst": {
            "x_axis": "categorical",
            "y_axis": "numeric",
            "color": True,
            "description": "Sunburst chart for hierarchical data"
        }
    }
    
    return options

def generate_visualization(data, viz_type, x=None, y=None, color=None, bins=20, title=None):
    """
    Generate a visualization based on the specified parameters.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to visualize
    viz_type : str
        The type of visualization to generate
    x : str or list, optional
        The column(s) to use for the x-axis
    y : str or list, optional
        The column(s) to use for the y-axis
    color : str, optional
        The column to use for color
    bins : int, optional
        The number of bins for histograms
    title : str, optional
        The title for the visualization
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The generated visualization
    """
    # Set default title if not provided
    if title is None:
        title = f"{viz_type.capitalize()} Plot"
    
    # Create the appropriate visualization based on the type
    if viz_type == "bar":
        if y is None:
            # If only x is provided, count occurrences
            fig = px.bar(data, x=x, title=title, color=color)
        else:
            fig = px.bar(data, x=x, y=y, title=title, color=color)
    
    elif viz_type == "line":
        fig = px.line(data, x=x, y=y, title=title, color=color)
    
    elif viz_type == "scatter":
        fig = px.scatter(data, x=x, y=y, title=title, color=color)
        
        # Add trendline if no color is specified
        if color is None:
            fig.update_layout(
                shapes=[{
                    'type': 'line',
                    'x0': data[x].min(),
                    'y0': np.polyval(np.polyfit(data[x], data[y], 1), data[x].min()),
                    'x1': data[x].max(),
                    'y1': np.polyval(np.polyfit(data[x], data[y], 1), data[x].max()),
                    'line': {
                        'color': 'red',
                        'width': 2,
                        'dash': 'dash'
                    }
                }]
            )
    
    elif viz_type == "histogram":
        fig = px.histogram(data, x=x, nbins=bins, title=title)
    
    elif viz_type == "box":
        fig = px.box(data, x=x, y=y, title=title, color=color)
    
    elif viz_type == "violin":
        fig = px.violin(data, x=x, y=y, title=title, color=color, box=True)
    
    elif viz_type == "pie":
        # For pie charts, x is categories and y is values
        if y is None:
            # If only x is provided, count occurrences
            value_counts = data[x].value_counts().reset_index()
            value_counts.columns = ['category', 'count']
            fig = px.pie(value_counts, names='category', values='count', title=title)
        else:
            # Use x for names and y for values
            # Aggregate the data
            agg_data = data.groupby(x)[y].sum().reset_index()
            fig = px.pie(agg_data, names=x, values=y, title=title)
    
    elif viz_type == "heatmap":
        # Create a cross-tabulation of the two categorical variables
        heatmap_data = pd.crosstab(data[y], data[x])
        fig = px.imshow(heatmap_data, title=title, color_continuous_scale='Viridis')
    
    elif viz_type == "correlation":
        # Calculate the correlation matrix
        if isinstance(x, list) and len(x) > 1:
            corr_matrix = data[x].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu_r',
                origin='lower',
                title=title or "Correlation Matrix"
            )
            
            # Add correlation values as text
            for i, row in enumerate(corr_matrix.index):
                for j, col in enumerate(corr_matrix.columns):
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text=f"{corr_matrix.iloc[i, j]:.2f}",
                        showarrow=False,
                        font=dict(color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
                    )
        else:
            # Not enough variables for correlation
            fig = go.Figure()
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text="Correlation requires at least 2 numeric variables",
                showarrow=False,
                font=dict(size=20)
            )
    
    elif viz_type == "area":
        fig = px.area(data, x=x, y=y, title=title, color=color)
    
    elif viz_type == "treemap":
        # For treemap, x is path (hierarchical categories) and y is values
        fig = px.treemap(data, path=[x], values=y, title=title, color=color)
    
    elif viz_type == "sunburst":
        # For sunburst, x is path (hierarchical categories) and y is values
        fig = px.sunburst(data, path=[x], values=y, title=title, color=color)
    
    else:
        # Unsupported visualization type
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"Unsupported visualization type: {viz_type}",
            showarrow=False,
            font=dict(size=20)
        )
    
    # Update layout for better appearance
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    # Add source reference if any
    fig.update_layout(
        annotations=[
            dict(
                text="Source: DevInsight Data Analysis Tool",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=1,
                y=0,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=10)
            )
        ]
    )
    
    return fig

def get_visualization_as_base64(fig):
    """
    Convert a plotly figure to base64 encoded image.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The figure to convert
        
    Returns:
    --------
    str
        Base64 encoded image
    """
    img_bytes = fig.to_image(format="png")
    encoded = base64.b64encode(img_bytes).decode('ascii')
    return encoded
