from flask import Flask, request, jsonify, abort, send_file
import pandas as pd
import numpy as np
import json
import io
import base64
import traceback
from data_processor import process_data, get_data_summary, export_data
from visualizer import generate_visualization, get_visualization_as_base64
from analyzer import perform_statistical_analysis, detect_trends
import os

# Initialize Flask app
app = Flask(__name__)

# Global variable to store the data
DATA = None
FILENAME = None

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "DevInsight API is running",
        "endpoints": [
            "/api/data",
            "/api/summary",
            "/api/visualize",
            "/api/analyze",
            "/api/upload",
            "/api/trends"
        ]
    })

@app.route('/api/data', methods=['GET'])
def get_data():
    global DATA
    
    if DATA is None:
        return jsonify({"error": "No data available. Please upload data first."}), 404
    
    # Return the data as JSON
    records = DATA.to_dict(orient='records')
    return jsonify({
        "filename": FILENAME,
        "rows": len(DATA),
        "columns": list(DATA.columns),
        "data": records
    })

@app.route('/api/summary', methods=['GET'])
def get_summary():
    global DATA
    
    if DATA is None:
        return jsonify({"error": "No data available. Please upload data first."}), 404
    
    # Get columns parameter from query string
    columns_param = request.args.get('columns')
    if columns_param:
        # Split by comma and strip whitespace
        columns = [col.strip() for col in columns_param.split(',')]
        # Filter to only include columns that exist in the data
        columns = [col for col in columns if col in DATA.columns]
    else:
        columns = None
    
    try:
        # Get data summary
        summary = get_data_summary(DATA, columns)
        
        # Convert to JSON
        summary_json = summary.reset_index().to_dict(orient='records')
        return jsonify({
            "filename": FILENAME,
            "summary": summary_json
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/visualize', methods=['POST'])
def visualize():
    global DATA
    
    if DATA is None:
        return jsonify({"error": "No data available. Please upload data first."}), 404
    
    # Get visualization parameters from request
    params = request.json
    
    if not params:
        return jsonify({"error": "No parameters provided"}), 400
    
    try:
        # Required parameters
        viz_type = params.get('viz_type')
        if not viz_type:
            return jsonify({"error": "viz_type parameter is required"}), 400
        
        # Optional parameters
        x = params.get('x')
        y = params.get('y')
        color = params.get('color')
        bins = params.get('bins', 20)
        title = params.get('title')
        
        # Generate the visualization
        fig = generate_visualization(
            DATA,
            viz_type=viz_type,
            x=x,
            y=y,
            color=color,
            bins=bins,
            title=title
        )
        
        # Convert to base64
        img_base64 = get_visualization_as_base64(fig)
        
        return jsonify({
            "image": img_base64,
            "viz_type": viz_type,
            "params": params
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    global DATA
    
    if DATA is None:
        return jsonify({"error": "No data available. Please upload data first."}), 404
    
    # Get analysis parameters from request
    params = request.json
    
    if not params:
        return jsonify({"error": "No parameters provided"}), 400
    
    try:
        # Required parameters
        analysis_type = params.get('analysis_type')
        if not analysis_type:
            return jsonify({"error": "analysis_type parameter is required"}), 400
        
        # Optional parameters
        columns = params.get('columns')
        x = params.get('x')
        y = params.get('y')
        date_column = params.get('date_column')
        value_column = params.get('value_column')
        method = params.get('method', 'pearson')
        
        # Perform the analysis
        result = perform_statistical_analysis(
            DATA,
            analysis_type=analysis_type,
            columns=columns,
            x=x,
            y=y,
            date_column=date_column,
            value_column=value_column,
            method=method
        )
        
        # Process the result based on analysis type
        if analysis_type == "descriptive" or analysis_type == "correlation":
            # Convert DataFrame to dict
            result_json = result.reset_index().to_dict(orient='records')
            return jsonify({
                "analysis_type": analysis_type,
                "result": result_json
            })
        elif analysis_type == "regression":
            # Extract relevant information from regression result
            regression_result = {
                "params": result["params"],
                "r_squared": result["r_squared"],
                "adj_r_squared": result["adj_r_squared"],
                "f_statistic": result["f_statistic"],
                "p_value": result["p_value"]
            }
            return jsonify({
                "analysis_type": analysis_type,
                "result": regression_result
            })
        elif analysis_type == "time_series":
            # Process time series results
            ts_result = {}
            
            if "stationarity_test" in result:
                ts_result["stationarity_test"] = result["stationarity_test"]
            
            if "seasonality_test" in result:
                ts_result["seasonality_test"] = result["seasonality_test"]
            
            # We can't include the plots directly in the API response
            # Instead, we'll indicate their presence
            if "original_series" in result:
                ts_result["has_time_series_plot"] = True
            
            if "decomposition" in result:
                ts_result["has_decomposition_plot"] = True
            
            if "autocorrelation" in result:
                ts_result["has_acf_pacf_plot"] = True
            
            return jsonify({
                "analysis_type": analysis_type,
                "result": ts_result
            })
        else:
            return jsonify({
                "error": f"Unsupported analysis type: {analysis_type}"
            }), 400
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload():
    global DATA, FILENAME
    
    # Check if a file is included in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Get file extension
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        
        # Read the file based on its extension
        if file_extension == 'csv':
            data = pd.read_csv(file)
        elif file_extension == 'xlsx':
            data = pd.read_excel(file)
        elif file_extension == 'json':
            content = file.read().decode('utf-8')
            data = pd.DataFrame(json.loads(content))
        else:
            return jsonify({"error": f"Unsupported file type: {file_extension}"}), 400
        
        # Process the data
        DATA = process_data(data)
        FILENAME = file.filename
        
        return jsonify({
            "message": f"Successfully uploaded {file.filename}",
            "rows": len(DATA),
            "columns": list(DATA.columns)
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/trends', methods=['POST'])
def trends():
    global DATA
    
    if DATA is None:
        return jsonify({"error": "No data available. Please upload data first."}), 404
    
    # Get parameters from request
    params = request.json
    
    if not params:
        return jsonify({"error": "No parameters provided"}), 400
    
    try:
        # Required parameter
        target_column = params.get('target_column')
        if not target_column:
            return jsonify({"error": "target_column parameter is required"}), 400
        
        # Optional parameter
        window = params.get('window', 5)
        
        # Detect trends
        result = detect_trends(
            DATA,
            target_column=target_column,
            window=window
        )
        
        # Process the result
        trend_result = {
            "mann_kendall": result["mann_kendall"],
            "has_trend_plot": "trend_plot" in result,
            "has_outlier_plot": "outlier_plot" in result
        }
        
        if "outliers" in result:
            trend_result["outlier_count"] = result["outliers"].sum()
        
        return jsonify({
            "target_column": target_column,
            "window": window,
            "result": trend_result
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=8000, debug=True)
