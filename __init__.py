"""
Radar Chart Visualization Package

A comprehensive package for creating beautiful and customizable radar charts
for data visualization, particularly useful for cluster analysis and 
multi-dimensional data comparison.
"""

from .radar_chart import RadarChart
from .cluster_radar import ClusterRadarChart
from .utils import normalize_data, calculate_cluster_stats

__version__ = "1.0.0"
__author__ = "Eugene Kim"

__all__ = [
    'RadarChart',
    'ClusterRadarChart', 
    'normalize_data',
    'calculate_cluster_stats'
] 