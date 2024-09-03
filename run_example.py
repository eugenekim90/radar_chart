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