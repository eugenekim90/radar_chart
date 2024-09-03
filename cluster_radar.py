#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ClusterRadarChart class for creating radar charts from cluster analysis data.

This module provides functionality to create radar charts from both raw data
with cluster labels and pre-aggregated cluster data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import seaborn as sns
from radar_chart import RadarChart
from utils import (normalize_data, calculate_cluster_stats, get_numeric_features, 
                   validate_data, prepare_radar_data)


class ClusterRadarChart(RadarChart):
    """
    A specialized radar chart class for cluster analysis visualization.
    
    This class can handle both:
    1. Raw data with cluster labels (will aggregate automatically)
    2. Pre-aggregated cluster data (ready for visualization)
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (18, 12),
                 dpi: int = 300,
                 style: str = 'seaborn-v0_8'):
        """
        Initialize the ClusterRadarChart.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default (18, 12)
            Figure size in inches (width, height) - larger for grid layout
        dpi : int, default 300
            Resolution for saved figures
        style : str, default 'seaborn-v0_8'
            Matplotlib style to use
        """
        super().__init__(figsize, dpi, style)
    
    def create_cluster_radar_grid(self,
                                 data: pd.DataFrame,
                                 title: str = "Cluster Analysis Radar Charts",
                                 is_aggregated: bool = False,
                                 cluster_col: str = 'final_cluster',
                                 id_cols: List[str] = None,
                                 features: List[str] = None,
                                 max_features: int = 8,
                                 normalization: str = 'minmax',
                                 aggregation: str = 'mean',
                                 charts_per_row: int = 3,
                                 show_values: bool = True,
                                 show_cluster_stats: bool = True,
                                 color_palette: str = 'default',
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a grid of radar charts for cluster analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe - either raw data with cluster labels or pre-aggregated
        title : str, default "Cluster Analysis Radar Charts"
            Main title for the visualization
        is_aggregated : bool, default False
            True if data is already aggregated by clusters, False if raw data
        cluster_col : str, default 'final_cluster'
            Name of the cluster column (ignored if is_aggregated=True)
        id_cols : List[str], optional
            List of ID columns to exclude (e.g., ['uid', 'id', 'customer_id'])
        features : List[str], optional
            Specific features to use. If None, auto-detects numeric features
        max_features : int, default 8
            Maximum number of features to display for readability
        normalization : str, default 'minmax'
            Normalization method: 'minmax', 'standard', 'robust', 'none'
        aggregation : str, default 'mean'
            Aggregation method for raw data: 'mean', 'median', 'max', 'min'
        charts_per_row : int, default 3
            Number of charts per row in the grid
        show_values : bool, default True
            Whether to show value labels on each chart
        show_cluster_stats : bool, default True
            Whether to show cluster size and percentage in titles
        color_palette : str, default 'default'
            Color palette to use for the charts
        save_path : Optional[str], default None
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Set up ID columns to exclude
        if id_cols is None:
            id_cols = ['uid', 'id', 'customer_id', 'user_id', 'index']
        
        # Process the data based on whether it's aggregated or not
        if is_aggregated:
            processed_data, cluster_stats = self._process_aggregated_data(
                data, features, id_cols, max_features, normalization
            )
        else:
            processed_data, cluster_stats = self._process_raw_data(
                data, cluster_col, features, id_cols, max_features, 
                normalization, aggregation
            )
        
        # Get final features and clusters
        final_features = list(processed_data.columns)
        clusters = processed_data.index.tolist()
        
        print(f"Creating radar charts for {len(clusters)} clusters with {len(final_features)} features")
        print(f"Features: {final_features}")
        
        # Set up the grid layout
        num_clusters = len(clusters)
        rows = (num_clusters + charts_per_row - 1) // charts_per_row
        
        # Create the figure and subplots
        fig, axs = plt.subplots(rows, charts_per_row, figsize=self.figsize, 
                               subplot_kw=dict(polar=True))
        
        # Handle single row case
        if rows == 1:
            if num_clusters == 1:
                axs = [axs]
            else:
                axs = np.array([axs])
        axs = axs.flatten()
        
        # Set up colors
        colors = self.color_palettes[color_palette][:num_clusters]
        
        # Calculate angles for radar chart
        num_vars = len(final_features)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        # Create radar chart for each cluster
        for i, cluster in enumerate(clusters):
            # Get values for this cluster
            values = processed_data.loc[cluster].tolist()
            values += values[:1]  # Close the circle
            
            # Plot the radar chart
            color = colors[i % len(colors)]
            axs[i].fill(angles, values, alpha=0.25, color=color)
            axs[i].plot(angles, values, color=color, linewidth=2, marker='o', markersize=6)
            
            # Customize the chart
            axs[i].set_ylim(0, 1)
            axs[i].set_yticklabels([])
            axs[i].set_xticks(angles[:-1])
            axs[i].set_xticklabels(final_features, fontsize=9)
            axs[i].grid(True, alpha=0.3)
            
            # Add value labels if requested
            if show_values:
                for j in range(num_vars):
                    axs[i].text(angles[j], values[j] + 0.05, f'{values[j]:.2f}', 
                               horizontalalignment='center', size=8, 
                               color='black', weight='semibold')
            
            # Create title with cluster information
            if show_cluster_stats and cluster_stats and cluster in cluster_stats:
                stats = cluster_stats[cluster]
                title_text = f"Cluster {cluster}: {stats['size']} points ({stats['percentage']:.1f}%)"
            else:
                title_text = f"Cluster {cluster}"
            
            axs[i].set_title(title_text, fontsize=12, fontweight='bold', pad=15)
        
        # Remove unused subplots
        for i in range(num_clusters, len(axs)):
            fig.delaxes(axs[i])
        
        # Add main title and layout adjustments
        plt.suptitle(title, size=20, color='blue', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Visualization saved as '{save_path}'")
        
        return fig
    
    def _process_raw_data(self, 
                         data: pd.DataFrame,
                         cluster_col: str,
                         features: List[str],
                         id_cols: List[str],
                         max_features: int,
                         normalization: str,
                         aggregation: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Process raw data with cluster labels.
        
        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            Processed aggregated data and cluster statistics
        """
        # Validate the data
        if cluster_col not in data.columns:
            raise ValueError(f"Cluster column '{cluster_col}' not found in data")
        
        # Get features if not specified
        if features is None:
            exclude_cols = id_cols + [cluster_col]
            features = get_numeric_features(data, exclude_cols)
        
        # Limit features for readability
        if len(features) > max_features:
            print(f"Limiting to first {max_features} features for readability")
            features = features[:max_features]
        
        # Validate data
        validate_data(data, features, cluster_col)
        
        # Calculate cluster statistics
        cluster_stats = calculate_cluster_stats(data, cluster_col)
        
        # Prepare aggregated data
        agg_data = prepare_radar_data(data, features, cluster_col, aggregation)
        
        # Normalize if requested
        if normalization != 'none':
            # Convert to DataFrame for normalization
            temp_df = agg_data.reset_index()
            normalized_df = normalize_data(temp_df, features, normalization)
            agg_data = normalized_df.set_index(cluster_col)[features]
        
        return agg_data, cluster_stats
    
    def _process_aggregated_data(self,
                                data: pd.DataFrame,
                                features: List[str],
                                id_cols: List[str],
                                max_features: int,
                                normalization: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Process pre-aggregated cluster data.
        
        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            Processed data and empty cluster statistics
        """
        # Get features if not specified
        if features is None:
            features = get_numeric_features(data, id_cols)
        
        # Limit features for readability
        if len(features) > max_features:
            print(f"Limiting to first {max_features} features for readability")
            features = features[:max_features]
        
        # Check if features exist
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")
        
        # Extract feature data
        feature_data = data[features].copy()
        
        # Normalize if requested
        if normalization != 'none':
            feature_data = normalize_data(feature_data, features, normalization)
        
        # Return data with empty cluster stats (since we don't have raw data)
        return feature_data, {}
    
    def create_single_cluster_radar(self,
                                   data: pd.DataFrame,
                                   cluster_id: Union[int, str],
                                   title: Optional[str] = None,
                                   is_aggregated: bool = False,
                                   cluster_col: str = 'final_cluster',
                                   id_cols: List[str] = None,
                                   features: List[str] = None,
                                   normalization: str = 'minmax',
                                   aggregation: str = 'mean',
                                   color: str = '#1f77b4',
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a radar chart for a single cluster.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
        cluster_id : Union[int, str]
            ID of the cluster to visualize
        title : Optional[str], default None
            Chart title. If None, auto-generates based on cluster_id
        is_aggregated : bool, default False
            Whether data is pre-aggregated
        cluster_col : str, default 'final_cluster'
            Name of cluster column (for raw data)
        id_cols : List[str], optional
            ID columns to exclude
        features : List[str], optional
            Features to include
        normalization : str, default 'minmax'
            Normalization method
        aggregation : str, default 'mean'
            Aggregation method for raw data
        color : str, default '#1f77b4'
            Color for the radar chart
        save_path : Optional[str], default None
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Set up ID columns
        if id_cols is None:
            id_cols = ['uid', 'id', 'customer_id', 'user_id', 'index']
        
        # Process data
        if is_aggregated:
            if cluster_id not in data.index:
                raise ValueError(f"Cluster {cluster_id} not found in aggregated data")
            
            if features is None:
                features = get_numeric_features(data, id_cols)
            
            cluster_data = data.loc[cluster_id, features]
            
            if normalization != 'none':
                temp_df = pd.DataFrame([cluster_data], columns=features)
                normalized_df = normalize_data(temp_df, features, normalization)
                values = normalized_df.iloc[0].tolist()
            else:
                values = cluster_data.tolist()
        else:
            # Filter data for specific cluster
            cluster_mask = data[cluster_col] == cluster_id
            if not cluster_mask.any():
                raise ValueError(f"Cluster {cluster_id} not found in data")
            
            cluster_subset = data[cluster_mask]
            
            if features is None:
                exclude_cols = id_cols + [cluster_col]
                features = get_numeric_features(data, exclude_cols)
            
            # Aggregate the cluster data
            if aggregation == 'mean':
                cluster_values = cluster_subset[features].mean()
            elif aggregation == 'median':
                cluster_values = cluster_subset[features].median()
            elif aggregation == 'max':
                cluster_values = cluster_subset[features].max()
            elif aggregation == 'min':
                cluster_values = cluster_subset[features].min()
            
            # Normalize if requested
            if normalization != 'none':
                temp_df = pd.DataFrame([cluster_values], columns=features)
                normalized_df = normalize_data(temp_df, features, normalization)
                values = normalized_df.iloc[0].tolist()
            else:
                values = cluster_values.tolist()
        
        # Set title
        if title is None:
            title = f"Cluster {cluster_id} Radar Chart"
        
        # Create the radar chart
        return self.create_single_radar(
            values=values,
            labels=features,
            title=title,
            color=color,
            save_path=save_path
        )
    
    def compare_clusters(self,
                        data: pd.DataFrame,
                        cluster_ids: List[Union[int, str]],
                        title: str = "Cluster Comparison",
                        is_aggregated: bool = False,
                        cluster_col: str = 'final_cluster',
                        id_cols: List[str] = None,
                        features: List[str] = None,
                        normalization: str = 'minmax',
                        aggregation: str = 'mean',
                        colors: Optional[List[str]] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comparison radar chart for multiple clusters.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
        cluster_ids : List[Union[int, str]]
            List of cluster IDs to compare
        title : str, default "Cluster Comparison"
            Chart title
        is_aggregated : bool, default False
            Whether data is pre-aggregated
        cluster_col : str, default 'final_cluster'
            Name of cluster column
        id_cols : List[str], optional
            ID columns to exclude
        features : List[str], optional
            Features to include
        normalization : str, default 'minmax'
            Normalization method
        aggregation : str, default 'mean'
            Aggregation method
        colors : Optional[List[str]], default None
            Colors for each cluster
        save_path : Optional[str], default None
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Set up ID columns
        if id_cols is None:
            id_cols = ['uid', 'id', 'customer_id', 'user_id', 'index']
        
        # Prepare data dictionary for comparison
        data_dict = {}
        
        for cluster_id in cluster_ids:
            if is_aggregated:
                if cluster_id not in data.index:
                    print(f"Warning: Cluster {cluster_id} not found in data, skipping...")
                    continue
                
                if features is None:
                    features = get_numeric_features(data, id_cols)
                
                cluster_data = data.loc[cluster_id, features]
                
                if normalization != 'none':
                    temp_df = pd.DataFrame([cluster_data], columns=features)
                    normalized_df = normalize_data(temp_df, features, normalization)
                    values = normalized_df.iloc[0].tolist()
                else:
                    values = cluster_data.tolist()
            else:
                cluster_mask = data[cluster_col] == cluster_id
                if not cluster_mask.any():
                    print(f"Warning: Cluster {cluster_id} not found in data, skipping...")
                    continue
                
                cluster_subset = data[cluster_mask]
                
                if features is None:
                    exclude_cols = id_cols + [cluster_col]
                    features = get_numeric_features(data, exclude_cols)
                
                # Aggregate
                if aggregation == 'mean':
                    cluster_values = cluster_subset[features].mean()
                elif aggregation == 'median':
                    cluster_values = cluster_subset[features].median()
                elif aggregation == 'max':
                    cluster_values = cluster_subset[features].max()
                elif aggregation == 'min':
                    cluster_values = cluster_subset[features].min()
                
                # Normalize
                if normalization != 'none':
                    temp_df = pd.DataFrame([cluster_values], columns=features)
                    normalized_df = normalize_data(temp_df, features, normalization)
                    values = normalized_df.iloc[0].tolist()
                else:
                    values = cluster_values.tolist()
            
            data_dict[f"Cluster {cluster_id}"] = values
        
        if not data_dict:
            raise ValueError("No valid clusters found for comparison")
        
        # Create comparison chart
        return self.create_comparison_radar(
            data_dict=data_dict,
            labels=features,
            title=title,
            colors=colors,
            save_path=save_path
        ) 