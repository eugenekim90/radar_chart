#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Examples demonstrating the radar chart package functionality.

This script shows how to use the radar chart package with both:
1. Raw data with cluster labels
2. Pre-aggregated cluster data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from radar_chart import RadarChart, ClusterRadarChart


def generate_sample_raw_data(n_samples=300, n_features=6, n_clusters=4):
    """
    Generate sample raw data with cluster labels for demonstration.
    
    Parameters
    ----------
    n_samples : int, default 300
        Number of data points
    n_features : int, default 6
        Number of features
    n_clusters : int, default 4
        Number of clusters
        
    Returns
    -------
    pd.DataFrame
        Sample dataframe with features and cluster labels
    """
    np.random.seed(42)
    
    # Generate cluster centers
    cluster_centers = np.random.rand(n_clusters, n_features) * 10
    
    data = []
    for i in range(n_samples):
        # Assign to random cluster
        cluster = np.random.randint(0, n_clusters)
        
        # Generate point around cluster center with some noise
        point = cluster_centers[cluster] + np.random.normal(0, 1.5, n_features)
        
        # Create row
        row = {
            'uid': f'user_{i:04d}',
            'feature_1': point[0],
            'feature_2': point[1], 
            'feature_3': point[2],
            'feature_4': point[3],
            'feature_5': point[4],
            'feature_6': point[5],
            'final_cluster': cluster
        }
        data.append(row)
    
    return pd.DataFrame(data)


def generate_sample_aggregated_data(n_clusters=4, n_features=6):
    """
    Generate sample pre-aggregated cluster data.
    
    Parameters
    ----------
    n_clusters : int, default 4
        Number of clusters
    n_features : int, default 6
        Number of features
        
    Returns
    -------
    pd.DataFrame
        Sample aggregated dataframe
    """
    np.random.seed(42)
    
    data = []
    for cluster in range(n_clusters):
        row = {
            'cluster_id': cluster,
            'avg_feature_1': np.random.rand() * 10,
            'avg_feature_2': np.random.rand() * 10,
            'avg_feature_3': np.random.rand() * 10,
            'avg_feature_4': np.random.rand() * 10,
            'avg_feature_5': np.random.rand() * 10,
            'avg_feature_6': np.random.rand() * 10,
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.set_index('cluster_id', inplace=True)
    return df


def example_1_raw_data_basic():
    """
    Example 1: Basic usage with raw data containing cluster labels.
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage with Raw Data")
    print("=" * 60)
    
    # Generate sample data
    raw_data = generate_sample_raw_data()
    print(f"Generated raw data with shape: {raw_data.shape}")
    print(f"Clusters: {sorted(raw_data['final_cluster'].unique())}")
    print(f"Features: {[col for col in raw_data.columns if col not in ['uid', 'final_cluster']]}")
    print()
    
    # Create radar chart
    radar = ClusterRadarChart()
    
    fig = radar.create_cluster_radar_grid(
        data=raw_data,
        title="Customer Segmentation Analysis",
        is_aggregated=False,  # This is raw data
        cluster_col='final_cluster',
        id_cols=['uid'],  # Columns to exclude
        save_path='example_1_raw_data.png'
    )
    
    plt.show()
    print("Example 1 completed! Check 'example_1_raw_data.png'\n")


def example_2_aggregated_data():
    """
    Example 2: Usage with pre-aggregated cluster data.
    """
    print("=" * 60)
    print("EXAMPLE 2: Usage with Pre-aggregated Data")
    print("=" * 60)
    
    # Generate sample aggregated data
    agg_data = generate_sample_aggregated_data()
    print(f"Generated aggregated data with shape: {agg_data.shape}")
    print(f"Clusters: {list(agg_data.index)}")
    print(f"Features: {list(agg_data.columns)}")
    print()
    
    # Create radar chart
    radar = ClusterRadarChart()
    
    fig = radar.create_cluster_radar_grid(
        data=agg_data,
        title="Pre-aggregated Cluster Analysis",
        is_aggregated=True,  # This is already aggregated
        save_path='example_2_aggregated.png'
    )
    
    plt.show()
    print("Example 2 completed! Check 'example_2_aggregated.png'\n")


def example_3_single_cluster():
    """
    Example 3: Visualizing a single cluster.
    """
    print("=" * 60)
    print("EXAMPLE 3: Single Cluster Visualization")
    print("=" * 60)
    
    # Generate sample data
    raw_data = generate_sample_raw_data()
    
    # Create radar chart for single cluster
    radar = ClusterRadarChart(figsize=(10, 10))
    
    fig = radar.create_single_cluster_radar(
        data=raw_data,
        cluster_id=2,
        title="Deep Dive: Cluster 2 Analysis",
        is_aggregated=False,
        cluster_col='final_cluster',
        id_cols=['uid'],
        color='#e74c3c',
        save_path='example_3_single_cluster.png'
    )
    
    plt.show()
    print("Example 3 completed! Check 'example_3_single_cluster.png'\n")


def example_4_cluster_comparison():
    """
    Example 4: Comparing multiple clusters.
    """
    print("=" * 60)
    print("EXAMPLE 4: Cluster Comparison")
    print("=" * 60)
    
    # Generate sample data
    raw_data = generate_sample_raw_data()
    
    # Compare specific clusters
    radar = ClusterRadarChart(figsize=(12, 10))
    
    fig = radar.compare_clusters(
        data=raw_data,
        cluster_ids=[0, 2, 3],
        title="Cluster Comparison: 0 vs 2 vs 3",
        is_aggregated=False,
        cluster_col='final_cluster',
        id_cols=['uid'],
        colors=['#3498db', '#e74c3c', '#2ecc71'],
        save_path='example_4_comparison.png'
    )
    
    plt.show()
    print("Example 4 completed! Check 'example_4_comparison.png'\n")


def example_5_custom_features_and_normalization():
    """
    Example 5: Custom features and different normalization methods.
    """
    print("=" * 60)
    print("EXAMPLE 5: Custom Features & Normalization")
    print("=" * 60)
    
    # Generate sample data
    raw_data = generate_sample_raw_data()
    
    # Create radar chart with custom settings
    radar = ClusterRadarChart()
    
    # Specify only certain features
    custom_features = ['feature_1', 'feature_3', 'feature_5']
    
    fig = radar.create_cluster_radar_grid(
        data=raw_data,
        title="Custom Features with Standard Normalization",
        is_aggregated=False,
        cluster_col='final_cluster',
        id_cols=['uid'],
        features=custom_features,  # Only use these features
        normalization='standard',  # Use standard normalization instead of minmax
        aggregation='median',  # Use median instead of mean
        color_palette='viridis',  # Different color palette
        save_path='example_5_custom.png'
    )
    
    plt.show()
    print("Example 5 completed! Check 'example_5_custom.png'\n")


def example_6_basic_radar_chart():
    """
    Example 6: Using the basic RadarChart class for general purposes.
    """
    print("=" * 60)
    print("EXAMPLE 6: Basic RadarChart for General Use")
    print("=" * 60)
    
    # Create a basic radar chart
    radar = RadarChart(figsize=(10, 10))
    
    # Sample data for a single entity
    values = [0.8, 0.6, 0.9, 0.4, 0.7, 0.5]
    labels = ['Speed', 'Accuracy', 'Efficiency', 'Cost', 'Quality', 'Reliability']
    
    fig = radar.create_single_radar(
        values=values,
        labels=labels,
        title="Performance Metrics Dashboard",
        color='#9b59b6',
        save_path='example_6_basic.png'
    )
    
    plt.show()
    print("Example 6 completed! Check 'example_6_basic.png'\n")


def example_7_advanced_radar():
    """
    Example 7: Advanced radar chart with special features.
    """
    print("=" * 60)
    print("EXAMPLE 7: Advanced Radar Chart Features")
    print("=" * 60)
    
    # Create an advanced radar chart
    radar = RadarChart(figsize=(12, 12))
    
    # Sample data
    values = [0.8, 0.6, 0.9, 0.4, 0.7, 0.5, 0.3, 0.8]
    labels = ['Speed', 'Accuracy', 'Efficiency', 'Cost', 'Quality', 'Reliability', 'Innovation', 'Scalability']
    
    fig = radar.create_advanced_radar(
        values=values,
        labels=labels,
        title="Advanced Performance Analysis",
        color='#e67e22',
        show_ranges=True,
        show_average=True,
        highlight_max=True,
        highlight_min=True,
        save_path='example_7_advanced.png'
    )
    
    plt.show()
    print("Example 7 completed! Check 'example_7_advanced.png'\n")


def run_all_examples():
    """
    Run all examples in sequence.
    """
    print("🎯 RADAR CHART PACKAGE EXAMPLES")
    print("=" * 60)
    print("This script demonstrates various ways to use the radar chart package.")
    print("Charts will be displayed and saved as PNG files.")
    print()
    
    try:
        example_1_raw_data_basic()
        example_2_aggregated_data()
        example_3_single_cluster()
        example_4_cluster_comparison()
        example_5_custom_features_and_normalization()
        example_6_basic_radar_chart()
        example_7_advanced_radar()
        
        print("🎉 All examples completed successfully!")
        print("Check the generated PNG files for the visualizations.")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    run_all_examples() 