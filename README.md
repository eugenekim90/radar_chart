# Radar Chart Visualization Package

A comprehensive Python package for creating beautiful and customizable radar charts, particularly useful for cluster analysis and multi-dimensional data comparison.

## Features

- **Flexible Data Input**: Works with both raw data (with cluster labels) and pre-aggregated data
- **Automatic Processing**: Handles data aggregation, normalization, and feature selection automatically
- **Multiple Chart Types**: Single radar, comparison radar, grid layout, and advanced radar with special features
- **Extensive Customization**: Colors, styles, normalization methods, aggregation functions, and more
- **Professional Output**: High-quality charts suitable for presentations and publications

## Installation

```bash
# Install required dependencies
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Quick Start

### Simple Example with Real Data

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple radar chart example with ccdata_with_clusters.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
from cluster_radar import ClusterRadarChart

def main():
    print("🚀 SIMPLE RADAR CHART EXAMPLE")
    print("=" * 40)
    
    # Load the data
    print("📊 Loading data...")
    df = pd.read_csv('ccdata_with_clusters.csv')
    print(f"✅ Loaded {df.shape[0]} customers with {len(df['final_cluster'].unique())} clusters")
    
    # Create one clean radar chart
    radar = ClusterRadarChart(figsize=(12, 8))
    
    fig = radar.create_cluster_radar_grid(
        data=df,
        title="Customer Segmentation Analysis",
        is_aggregated=False,
        cluster_col='final_cluster',
        id_cols=['CUST_ID'],
        max_features=6,  # Keep it simple
        save_path='radar_chart.png'
    )
    
    plt.show()
    print("✅ Done! Saved as: radar_chart.png")

if __name__ == "__main__":
    main()

### Sample Output

Running the above example generates this radar chart visualization:

![Customer Segmentation Radar Chart](https://raw.githubusercontent.com/eugenekim90/radar_chart/main/radar_chart.png)

*If the image doesn't display above, you can [view it directly here](https://github.com/eugenekim90/radar_chart/blob/main/radar_chart.png)*

This example:
- Loads customer data with 8,950 customers across 9 clusters
- Creates a clean radar chart grid showing all clusters
- Uses 6 key financial features for visualization
- Saves the result as a single PNG file

### Example 1: Raw Data with Cluster Labels

```python
from radar_charts import ClusterRadarChart
import pandas as pd

# Your data with cluster labels
# df should have numeric features and a cluster column
radar = ClusterRadarChart()

fig = radar.create_cluster_radar_grid(
    data=df,
    title="Customer Segmentation Analysis",
    is_aggregated=False,  # Raw data
    cluster_col='final_cluster',
    id_cols=['uid', 'customer_id'],  # Columns to exclude
    save_path='cluster_analysis.png'
)
```

### Example 2: Pre-aggregated Data

```python
# Your pre-aggregated data (clusters as rows, features as columns)
radar = ClusterRadarChart()

fig = radar.create_cluster_radar_grid(
    data=aggregated_df,
    title="Pre-aggregated Cluster Analysis", 
    is_aggregated=True,  # Already aggregated
    save_path='aggregated_analysis.png'
)
```

## Main Classes

### ClusterRadarChart

The main class for cluster analysis visualization. Handles both raw and aggregated data.

#### Key Methods:

- `create_cluster_radar_grid()`: Creates a grid of radar charts for all clusters
- `create_single_cluster_radar()`: Creates a radar chart for a single cluster
- `compare_clusters()`: Creates a comparison chart for multiple clusters

#### Key Parameters:

- `is_aggregated`: `True` if data is pre-aggregated, `False` if raw data with labels
- `cluster_col`: Name of the cluster column (for raw data)
- `id_cols`: List of ID columns to exclude from analysis
- `features`: Specific features to use (auto-detected if None)
- `normalization`: 'minmax', 'standard', 'robust', or 'none'
- `aggregation`: 'mean', 'median', 'max', or 'min' (for raw data)

## Running Examples

```bash
python run_example.py
```

This will load the ccdata_with_clusters.csv file and create a radar chart visualization.

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## License

This package is provided as-is for educational and research purposes.