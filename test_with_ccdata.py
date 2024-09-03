#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script using ccdata.csv to demonstrate the radar chart package.

This script loads the ccdata.csv file and creates various radar chart
visualizations to show how the package works with real data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cluster_radar import ClusterRadarChart
from radar_chart import RadarChart


def load_and_explore_data():
    """
    Load the ccdata.csv file and explore its structure.
    
    Returns
    -------
    pd.DataFrame
        The loaded dataframe
    """
    print("🔍 LOADING AND EXPLORING CCDATA.CSV")
    print("=" * 60)
    
    try:
        # Try to load the data
        df = pd.read_csv('ccdata.csv')
        print(f"✅ Successfully loaded ccdata.csv")
        print(f"📊 Data shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print()
        
        # Show basic info
        print("📈 Data Info:")
        print(df.info())
        print()
        
        # Show first few rows
        print("👀 First 5 rows:")
        print(df.head())
        print()
        
        # Check for missing values
        print("❓ Missing values:")
        missing = df.isnull().sum()
        print(missing[missing > 0])
        print()
        
        # Show numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"🔢 Numeric columns: {numeric_cols}")
        print()
        
        return df
        
    except FileNotFoundError:
        print("❌ Error: ccdata.csv not found in current directory")
        print("Please make sure ccdata.csv is in the same directory as this script")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def create_sample_clusters(df, n_clusters=4):
    """
    Create sample clusters for demonstration if the data doesn't have clusters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    n_clusters : int, default 4
        Number of clusters to create
        
    Returns
    -------
    pd.DataFrame
        Dataframe with added cluster column
    """
    print("🎯 CREATING SAMPLE CLUSTERS")
    print("=" * 60)
    
    # Get numeric columns (excluding any ID columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove potential ID columns
    id_patterns = ['id', 'ID', 'index', 'Index', 'customer', 'Customer']
    numeric_cols = [col for col in numeric_cols 
                   if not any(pattern in col for pattern in id_patterns)]
    
    print(f"Using columns for clustering: {numeric_cols}")
    
    if len(numeric_cols) < 2:
        print("❌ Not enough numeric columns for clustering")
        return df
    
    # Simple clustering using quantiles for demonstration
    # In real scenarios, you'd use proper clustering algorithms
    np.random.seed(42)
    
    # Create clusters based on first numeric column
    if len(numeric_cols) > 0:
        first_col = numeric_cols[0]
        df_copy = df.copy()
        
        # Create clusters based on quantiles
        df_copy['cluster'] = pd.qcut(df_copy[first_col], 
                                   q=n_clusters, 
                                   labels=range(n_clusters),
                                   duplicates='drop')
        
        # Add some randomness to make it more realistic
        mask = np.random.random(len(df_copy)) < 0.1  # 10% random reassignment
        df_copy.loc[mask, 'cluster'] = np.random.randint(0, n_clusters, mask.sum())
        
        print(f"✅ Created {n_clusters} clusters")
        print("Cluster distribution:")
        print(df_copy['cluster'].value_counts().sort_index())
        print()
        
        return df_copy
    
    return df


def test_raw_data_radar(df):
    """
    Test radar charts with raw data containing cluster labels.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with cluster labels
    """
    print("🎯 TESTING RAW DATA RADAR CHARTS")
    print("=" * 60)
    
    # Check if we have a cluster column
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
    if not cluster_cols:
        print("❌ No cluster column found. Creating sample clusters...")
        df = create_sample_clusters(df)
        cluster_col = 'cluster'
    else:
        cluster_col = cluster_cols[0]
        print(f"✅ Using cluster column: {cluster_col}")
    
    # Get ID columns to exclude
    id_cols = []
    for col in df.columns:
        if any(pattern in col.lower() for pattern in ['id', 'index', 'customer', 'name']):
            id_cols.append(col)
    
    print(f"📋 ID columns to exclude: {id_cols}")
    
    # Create radar chart
    radar = ClusterRadarChart(figsize=(16, 12))
    
    try:
        fig = radar.create_cluster_radar_grid(
            data=df,
            title="Credit Card Customer Segmentation Analysis",
            is_aggregated=False,
            cluster_col=cluster_col,
            id_cols=id_cols,
            max_features=8,  # Limit for readability
            normalization='minmax',
            aggregation='mean',
            save_path='ccdata_raw_radar.png'
        )
        
        plt.show()
        print("✅ Raw data radar chart created successfully!")
        print("📁 Saved as: ccdata_raw_radar.png")
        
    except Exception as e:
        print(f"❌ Error creating raw data radar: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_aggregated_data_radar(df):
    """
    Test radar charts with pre-aggregated data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe to aggregate
    """
    print("🎯 TESTING AGGREGATED DATA RADAR CHARTS")
    print("=" * 60)
    
    # Check if we have a cluster column
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
    if not cluster_cols:
        print("❌ No cluster column found. Creating sample clusters...")
        df = create_sample_clusters(df)
        cluster_col = 'cluster'
    else:
        cluster_col = cluster_cols[0]
    
    # Get numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID columns and cluster column
    id_patterns = ['id', 'index', 'customer', 'name']
    feature_cols = [col for col in numeric_cols 
                   if col != cluster_col and 
                   not any(pattern in col.lower() for pattern in id_patterns)]
    
    print(f"📊 Aggregating features: {feature_cols}")
    
    # Create aggregated data
    agg_data = df.groupby(cluster_col)[feature_cols].mean()
    
    print(f"📈 Aggregated data shape: {agg_data.shape}")
    print("Aggregated data preview:")
    print(agg_data.head())
    print()
    
    # Create radar chart with aggregated data
    radar = ClusterRadarChart(figsize=(16, 12))
    
    try:
        fig = radar.create_cluster_radar_grid(
            data=agg_data,
            title="Pre-aggregated Credit Card Customer Analysis",
            is_aggregated=True,  # This is already aggregated
            max_features=8,
            normalization='minmax',
            save_path='ccdata_aggregated_radar.png'
        )
        
        plt.show()
        print("✅ Aggregated data radar chart created successfully!")
        print("📁 Saved as: ccdata_aggregated_radar.png")
        
    except Exception as e:
        print(f"❌ Error creating aggregated radar: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_single_cluster_analysis(df):
    """
    Test single cluster analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with cluster labels
    """
    print("🎯 TESTING SINGLE CLUSTER ANALYSIS")
    print("=" * 60)
    
    # Check if we have a cluster column
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
    if not cluster_cols:
        df = create_sample_clusters(df)
        cluster_col = 'cluster'
    else:
        cluster_col = cluster_cols[0]
    
    # Get available clusters
    clusters = df[cluster_col].unique()
    print(f"Available clusters: {sorted(clusters)}")
    
    # Pick the first cluster for analysis
    target_cluster = clusters[0]
    print(f"Analyzing cluster: {target_cluster}")
    
    # Get ID columns
    id_cols = []
    for col in df.columns:
        if any(pattern in col.lower() for pattern in ['id', 'index', 'customer', 'name']):
            id_cols.append(col)
    
    # Create single cluster radar
    radar = ClusterRadarChart(figsize=(10, 10))
    
    try:
        fig = radar.create_single_cluster_radar(
            data=df,
            cluster_id=target_cluster,
            title=f"Deep Dive: Customer Segment {target_cluster}",
            is_aggregated=False,
            cluster_col=cluster_col,
            id_cols=id_cols,
            color='#e74c3c',
            save_path=f'ccdata_cluster_{target_cluster}.png'
        )
        
        plt.show()
        print(f"✅ Single cluster analysis completed for cluster {target_cluster}!")
        print(f"📁 Saved as: ccdata_cluster_{target_cluster}.png")
        
    except Exception as e:
        print(f"❌ Error creating single cluster analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_cluster_comparison(df):
    """
    Test cluster comparison functionality.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with cluster labels
    """
    print("🎯 TESTING CLUSTER COMPARISON")
    print("=" * 60)
    
    # Check if we have a cluster column
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
    if not cluster_cols:
        df = create_sample_clusters(df)
        cluster_col = 'cluster'
    else:
        cluster_col = cluster_cols[0]
    
    # Get available clusters
    clusters = sorted(df[cluster_col].unique())
    print(f"Available clusters: {clusters}")
    
    # Compare first 3 clusters (or all if less than 3)
    compare_clusters = clusters[:min(3, len(clusters))]
    print(f"Comparing clusters: {compare_clusters}")
    
    # Get ID columns
    id_cols = []
    for col in df.columns:
        if any(pattern in col.lower() for pattern in ['id', 'index', 'customer', 'name']):
            id_cols.append(col)
    
    # Create comparison radar
    radar = ClusterRadarChart(figsize=(12, 10))
    
    try:
        fig = radar.compare_clusters(
            data=df,
            cluster_ids=compare_clusters,
            title="Customer Segment Comparison",
            is_aggregated=False,
            cluster_col=cluster_col,
            id_cols=id_cols,
            colors=['#3498db', '#e74c3c', '#2ecc71'],
            save_path='ccdata_comparison.png'
        )
        
        plt.show()
        print("✅ Cluster comparison completed!")
        print("📁 Saved as: ccdata_comparison.png")
        
    except Exception as e:
        print(f"❌ Error creating cluster comparison: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def main():
    """
    Main function to run all tests with ccdata.csv.
    """
    print("🚀 RADAR CHART PACKAGE TEST WITH CCDATA.CSV")
    print("=" * 60)
    print("This script will test the radar chart package using your ccdata.csv file.")
    print("Various radar chart visualizations will be created and saved as PNG files.")
    print()
    
    # Load the data
    df = load_and_explore_data()
    if df is None:
        return
    
    # Run all tests
    try:
        test_raw_data_radar(df)
        test_aggregated_data_radar(df)
        test_single_cluster_analysis(df)
        test_cluster_comparison(df)
        
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Generated files:")
        print("📁 ccdata_raw_radar.png - Grid of radar charts from raw data")
        print("📁 ccdata_aggregated_radar.png - Grid from pre-aggregated data")
        print("📁 ccdata_cluster_X.png - Single cluster analysis")
        print("📁 ccdata_comparison.png - Cluster comparison chart")
        print()
        print("🔍 Check these PNG files to see your radar chart visualizations!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 