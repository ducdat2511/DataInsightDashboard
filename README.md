# DevInsight - Data Analysis Tool

DevInsight is a powerful data analysis tool designed specifically for developers and technical teams that helps discover insights and trends from their data while offering technical integration capabilities.

![DevInsight](generated-icon.png)

## 🌟 Features

- **Versatile Data Import**: Import data from CSV, Excel, JSON files, or directly from APIs
- **Interactive Data Exploration**: Filter, query, and explore your datasets through an intuitive interface
- **Advanced Visualizations**: Create insightful charts including bar charts, scatter plots, histograms, heatmaps, and more
- **Statistical Analysis**: Perform statistical analyses including correlation, regression, and time-series analysis
- **Trend Detection**: Automatically detect trends and anomalies in your data
- **API Integration**: Connect your applications to DevInsight through a comprehensive REST API
- **Export Functionality**: Export your data and insights in various formats

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Required Python packages are listed in `pyproject.toml`

### Running the Application

1. Ensure you have the required dependencies installed
2. Start the application using Streamlit:

```bash
streamlit run app.py
```

3. Open your browser and go to http://localhost:5000

## 📊 Using DevInsight

### Data Import

1. Navigate to the "Data Import" page
2. Upload your data file (CSV, Excel, or JSON)
3. Alternatively, import data directly from an API

### Data Exploration

- Filter your data based on column values
- Run custom queries using Python syntax
- View summary statistics and correlation analysis

### Visualization

- Create various types of visualizations based on your data
- Customize charts with titles, colors, and axis labels
- Download visualizations for use in reports or presentations

### Analysis

- Perform statistical analyses including:
  - Descriptive statistics
  - Correlation analysis
  - Regression analysis
  - Time-series analysis
- Detect trends and anomalies in your data

### API Integration

DevInsight provides a REST API that allows you to:

- Access your data programmatically
- Generate visualizations
- Perform analysis
- Detect trends

API documentation is available on the "API Integration" page within the application.

## 🧰 Technical Architecture

DevInsight is built with the following components:

- **Frontend**: Streamlit for the interactive web interface
- **Data Processing**: Pandas and NumPy
- **Visualization**: Plotly and Matplotlib
- **Analysis**: SciPy, StatsModels, and Scikit-learn
- **API**: Flask REST API

## 📄 File Structure

- `app.py`: Main Streamlit application
- `data_processor.py`: Functions for data processing and summary
- `visualizer.py`: Functions for data visualization
- `analyzer.py`: Functions for statistical analysis and trend detection
- `api.py`: Flask REST API implementation
- `.streamlit/config.toml`: Streamlit configuration

## 🔌 API Documentation

The DevInsight API runs on port 8000 and provides the following endpoints:

- `GET /api/data`: Get the current dataset
- `GET /api/summary`: Get summary statistics
- `POST /api/visualize`: Generate visualizations
- `POST /api/analyze`: Perform statistical analysis
- `POST /api/upload`: Upload data
- `POST /api/trends`: Detect trends in the data

For detailed API documentation, please refer to the "API Integration" page in the application.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The DevInsight team
- All open-source libraries that make this tool possible