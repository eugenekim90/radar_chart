#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for radar chart data processing and analysis.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Dict, List, Tuple, Optional, Union


def normalize_data(data: pd.DataFrame, 
                  features: List[str], 
                  method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize feature data using various scaling methods.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing the features to normalize
    features : List[str]
        List of feature column names to normalize
    method : str, default 'minmax'
        Normalization method: 'minmax', 'standard', 'robust'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized features
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Method must be 'minmax', 'standard', or 'robust'")
    
    # Create a copy to avoid modifying original data
    normalized_data = data.copy()
    
    # Normalize only the specified features
    normalized_data[features] = scaler.fit_transform(data[features])
    
    return normalized_data


def calculate_cluster_stats(data: pd.DataFrame, 
                          cluster_col: str = 'final_cluster') -> Dict:
    """
    Calculate statistics for each cluster in the dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing cluster assignments
    cluster_col : str, default 'final_cluster'
        Name of the column containing cluster assignments
        
    Returns
    -------
    Dict
        Dictionary with cluster statistics including size and percentage
    """
    clusters = data[cluster_col].unique()
    total_points = len(data)
    
    cluster_stats = {}
    for cluster in clusters:
        size = len(data[data[cluster_col] == cluster])
        percentage = (size / total_points) * 100
        cluster_stats[cluster] = {
            'size': size, 
            'percentage': percentage,
            'cluster_id': cluster
        }
    
    return cluster_stats


def get_numeric_features(data: pd.DataFrame, 
                        exclude_cols: List[str] = None) -> List[str]:
    """
    Extract numeric feature columns from dataframe.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    exclude_cols : List[str], optional
        List of column names to exclude
        
    Returns
    -------
    List[str]
        List of numeric feature column names
    """
    if exclude_cols is None:
        exclude_cols = ['uid', 'final_cluster', 'cluster', 'id']
    
    features = [col for col in data.columns 
               if col not in exclude_cols and 
               pd.api.types.is_numeric_dtype(data[col])]
    
    return features


def calculate_feature_importance(data: pd.DataFrame, 
                               features: List[str],
                               cluster_col: str = 'final_cluster') -> pd.DataFrame:
    """
    Calculate feature importance based on variance between clusters.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    features : List[str]
        List of feature names
    cluster_col : str, default 'final_cluster'
        Name of cluster column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature importance scores
    """
    importance_scores = []
    
    for feature in features:
        # Calculate variance between cluster means
        cluster_means = data.groupby(cluster_col)[feature].mean()
        overall_mean = data[feature].mean()
        
        # Between-cluster variance
        between_var = np.var(cluster_means)
        
        # Within-cluster variance
        within_var = data.groupby(cluster_col)[feature].var().mean()
        
        # F-ratio as importance score
        if within_var > 0:
            importance = between_var / within_var
        else:
            importance = 0
            
        importance_scores.append({
            'feature': feature,
            'importance': importance,
            'between_var': between_var,
            'within_var': within_var
        })
    
    importance_df = pd.DataFrame(importance_scores)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


def prepare_radar_data(data: pd.DataFrame,
                      features: List[str],
                      cluster_col: str = 'final_cluster',
                      aggregation: str = 'mean') -> pd.DataFrame:
    """
    Prepare data for radar chart visualization.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    features : List[str]
        List of feature names
    cluster_col : str, default 'final_cluster'
        Name of cluster column
    aggregation : str, default 'mean'
        Aggregation method: 'mean', 'median', 'max', 'min'
        
    Returns
    -------
    pd.DataFrame
        Prepared data with cluster aggregations
    """
    if aggregation == 'mean':
        agg_data = data.groupby(cluster_col)[features].mean()
    elif aggregation == 'median':
        agg_data = data.groupby(cluster_col)[features].median()
    elif aggregation == 'max':
        agg_data = data.groupby(cluster_col)[features].max()
    elif aggregation == 'min':
        agg_data = data.groupby(cluster_col)[features].min()
    else:
        raise ValueError("Aggregation must be 'mean', 'median', 'max', or 'min'")
    
    return agg_data


def validate_data(data: pd.DataFrame, 
                 features: List[str],
                 cluster_col: str = 'final_cluster') -> bool:
    """
    Validate input data for radar chart creation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    features : List[str]
        List of feature names
    cluster_col : str, default 'final_cluster'
        Name of cluster column
        
    Returns
    -------
    bool
        True if data is valid, raises ValueError otherwise
    """
    # Check if dataframe is not empty
    if data.empty:
        raise ValueError("Input dataframe is empty")
    
    # Check if cluster column exists
    if cluster_col not in data.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in data")
    
    # Check if features exist
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Features not found in data: {missing_features}")
    
    # Check if features are numeric
    non_numeric = [f for f in features if not pd.api.types.is_numeric_dtype(data[f])]
    if non_numeric:
        raise ValueError(f"Non-numeric features found: {non_numeric}")
    
    # Check if there are any clusters
    if data[cluster_col].nunique() == 0:
        raise ValueError("No clusters found in data")
    
    return True 