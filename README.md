# Radar Chart Visualization Package

A comprehensive Python package for creating beautiful and customizable radar charts, particularly useful for cluster analysis and multi-dimensional data comparison.

## Quick Start

```python
import pandas as pd
import matplotlib.pyplot as plt
from cluster_radar import ClusterRadarChart

# Load your data
df = pd.read_csv('ccdata_with_clusters.csv')

# Create radar chart
radar = ClusterRadarChart(figsize=(12, 8))
fig = radar.create_cluster_radar_grid(
    data=df,
    title="Customer Segmentation Analysis",
    is_aggregated=False,
    cluster_col='final_cluster',
    id_cols=['CUST_ID'],
    max_features=6,
    save_path='radar_chart.png'
)
plt.show()
```

## Sample Output

![radar_chart](./radar_chart.png)

## Feature Selection Options

### Option 1: Specify Exact Features (Recommended)
```python
# Use specific columns you want to visualize
selected_features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'TENURE']

radar.create_cluster_radar_grid(
    data=df,
    features=selected_features,  # Specify which columns to use
    # ... other parameters
)
```

### Option 2: Auto-select with Limit
```python
# Let the system auto-detect features but limit the number
radar.create_cluster_radar_grid(
    data=df,
    features=None,        # Auto-detect all numeric features
    max_features=6,       # Limit to first 6 features
    # ... other parameters
)
```

### Option 3: Use All Available Features
```python
# Use all numeric features (not recommended for many features)
radar.create_cluster_radar_grid(
    data=df,
    features=None,        # Auto-detect all numeric features
    max_features=20,      # Set high limit or remove parameter
    # ... other parameters
)
```

## Key Parameters

- **`features`**: List of column names to use. If `None`, auto-detects numeric columns
- **`max_features`**: Maximum number of features to display (used when `features=None`)
- **`id_cols`**: List of ID columns to exclude (e.g., `['CUST_ID', 'user_id']`)
- **`cluster_col`**: Name of the cluster column (for raw data)
- **`is_aggregated`**: `True` if data is pre-aggregated, `False` if raw data

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

Run the example:
```bash
python run_example.py
```

## Features

- **Flexible Data Input**: Works with both raw data (with cluster labels) and pre-aggregated data
- **Column Selection**: Choose specific columns or auto-detect with limits
- **Automatic Processing**: Handles data aggregation, normalization, and feature selection automatically
- **Multiple Chart Types**: Single radar, comparison radar, grid layout, and advanced radar with special features
- **Extensive Customization**: Colors, styles, normalization methods, aggregation functions, and more
- **Professional Output**: High-quality charts suitable for presentations and publications

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn