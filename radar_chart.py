#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core RadarChart class for creating customizable radar/spider charts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import seaborn as sns


class RadarChart:
    """
    A flexible and customizable radar chart class for data visualization.
    
    This class provides extensive customization options for creating beautiful
    radar charts suitable for various data visualization needs.
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (10, 10),
                 dpi: int = 300,
                 style: str = 'seaborn-v0_8'):
        """
        Initialize the RadarChart.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default (10, 10)
            Figure size in inches (width, height)
        dpi : int, default 300
            Resolution for saved figures
        style : str, default 'seaborn-v0_8'
            Matplotlib style to use
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Set the style
        plt.style.use(self.style)
        
        # Color palettes
        self.color_palettes = {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'pastel': sns.color_palette("pastel", 10),
            'bright': sns.color_palette("bright", 10),
            'dark': sns.color_palette("dark", 10),
            'colorblind': sns.color_palette("colorblind", 10),
            'viridis': sns.color_palette("viridis", 10),
            'plasma': sns.color_palette("plasma", 10)
        }
    
    def create_single_radar(self,
                           values: List[float],
                           labels: List[str],
                           title: str = "Radar Chart",
                           color: str = '#1f77b4',
                           alpha: float = 0.3,
                           linewidth: float = 2,
                           marker: str = 'o',
                           markersize: float = 8,
                           show_values: bool = True,
                           value_format: str = '.2f',
                           grid_alpha: float = 0.3,
                           label_fontsize: int = 12,
                           title_fontsize: int = 16,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a single radar chart.
        
        Parameters
        ----------
        values : List[float]
            Values for each axis (should be normalized to 0-1 range)
        labels : List[str]
            Labels for each axis
        title : str, default "Radar Chart"
            Chart title
        color : str, default '#1f77b4'
            Color for the radar line and fill
        alpha : float, default 0.3
            Transparency for the filled area
        linewidth : float, default 2
            Width of the radar line
        marker : str, default 'o'
            Marker style for data points
        markersize : float, default 8
            Size of the markers
        show_values : bool, default True
            Whether to show value labels on the chart
        value_format : str, default '.2f'
            Format string for value labels
        grid_alpha : float, default 0.3
            Transparency for grid lines
        label_fontsize : int, default 12
            Font size for axis labels
        title_fontsize : int, default 16
            Font size for the title
        save_path : Optional[str], default None
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Validate inputs
        if len(values) != len(labels):
            raise ValueError("Values and labels must have the same length")
        
        # Calculate angles for each axis
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Close the radar chart
        values_closed = values + [values[0]]
        angles_closed = angles + [angles[0]]
        
        # Create the figure and polar subplot
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(polar=True))
        
        # Plot the radar chart
        ax.fill(angles_closed, values_closed, color=color, alpha=alpha)
        ax.plot(angles_closed, values_closed, color=color, linewidth=linewidth, 
                marker=marker, markersize=markersize)
        
        # Customize the chart
        ax.set_ylim(0, 1)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=label_fontsize)
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
        
        # Customize grid
        ax.grid(True, alpha=grid_alpha)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        
        # Add value labels if requested
        if show_values:
            for angle, value in zip(angles, values):
                ax.text(angle, value + 0.05, f'{value:{value_format}}',
                       horizontalalignment='center', fontsize=10,
                       fontweight='semibold')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        return fig
    
    def create_comparison_radar(self,
                              data_dict: Dict[str, List[float]],
                              labels: List[str],
                              title: str = "Radar Chart Comparison",
                              colors: Optional[List[str]] = None,
                              alpha: float = 0.2,
                              linewidth: float = 2,
                              show_legend: bool = True,
                              legend_loc: str = 'upper right',
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a radar chart comparing multiple datasets.
        
        Parameters
        ----------
        data_dict : Dict[str, List[float]]
            Dictionary with dataset names as keys and values as lists
        labels : List[str]
            Labels for each axis
        title : str, default "Radar Chart Comparison"
            Chart title
        colors : Optional[List[str]], default None
            Colors for each dataset. If None, uses default palette
        alpha : float, default 0.2
            Transparency for filled areas
        linewidth : float, default 2
            Width of radar lines
        show_legend : bool, default True
            Whether to show the legend
        legend_loc : str, default 'upper right'
            Legend location
        save_path : Optional[str], default None
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Validate inputs
        if not data_dict:
            raise ValueError("data_dict cannot be empty")
        
        # Check that all datasets have the same length as labels
        for name, values in data_dict.items():
            if len(values) != len(labels):
                raise ValueError(f"Dataset '{name}' length doesn't match labels length")
        
        # Set colors
        if colors is None:
            colors = self.color_palettes['default'][:len(data_dict)]
        elif len(colors) < len(data_dict):
            colors = colors * (len(data_dict) // len(colors) + 1)
        
        # Calculate angles
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles_closed = angles + [angles[0]]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(polar=True))
        
        # Plot each dataset
        for i, (name, values) in enumerate(data_dict.items()):
            values_closed = values + [values[0]]
            color = colors[i]
            
            ax.fill(angles_closed, values_closed, color=color, alpha=alpha, label=name)
            ax.plot(angles_closed, values_closed, color=color, linewidth=linewidth,
                   marker='o', markersize=6, label=name)
        
        # Customize the chart
        ax.set_ylim(0, 1)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Customize grid
        ax.grid(True, alpha=0.3)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        
        # Add legend
        if show_legend:
            plt.legend(loc=legend_loc, bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        return fig
    
    def create_advanced_radar(self,
                            values: List[float],
                            labels: List[str],
                            title: str = "Advanced Radar Chart",
                            color: str = '#1f77b4',
                            show_ranges: bool = True,
                            range_colors: List[str] = None,
                            show_average: bool = True,
                            average_color: str = 'red',
                            highlight_max: bool = True,
                            highlight_min: bool = True,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create an advanced radar chart with additional features.
        
        Parameters
        ----------
        values : List[float]
            Values for each axis (should be normalized to 0-1 range)
        labels : List[str]
            Labels for each axis
        title : str, default "Advanced Radar Chart"
            Chart title
        color : str, default '#1f77b4'
            Primary color for the radar
        show_ranges : bool, default True
            Whether to show colored range bands
        range_colors : List[str], optional
            Colors for range bands
        show_average : bool, default True
            Whether to show average line
        average_color : str, default 'red'
            Color for average line
        highlight_max : bool, default True
            Whether to highlight maximum value
        highlight_min : bool, default True
            Whether to highlight minimum value
        save_path : Optional[str], default None
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Calculate angles
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles_closed = angles + [angles[0]]
        values_closed = values + [values[0]]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(polar=True))
        
        # Add range bands if requested
        if show_ranges:
            if range_colors is None:
                range_colors = ['#ffcccc', '#ffffcc', '#ccffcc', '#ccffff', '#ccccff']
            
            for i, range_color in enumerate(range_colors):
                ax.fill_between(angles_closed, 
                              [i * 0.2] * len(angles_closed),
                              [(i + 1) * 0.2] * len(angles_closed),
                              color=range_color, alpha=0.3)
        
        # Plot main radar
        ax.fill(angles_closed, values_closed, color=color, alpha=0.3)
        ax.plot(angles_closed, values_closed, color=color, linewidth=3, 
                marker='o', markersize=8)
        
        # Show average line if requested
        if show_average:
            avg_value = np.mean(values)
            avg_line = [avg_value] * len(angles_closed)
            ax.plot(angles_closed, avg_line, color=average_color, 
                   linestyle='--', linewidth=2, alpha=0.8, label=f'Average: {avg_value:.2f}')
        
        # Highlight max and min values
        if highlight_max:
            max_idx = np.argmax(values)
            ax.scatter(angles[max_idx], values[max_idx], color='green', 
                      s=150, marker='^', zorder=5, label=f'Max: {values[max_idx]:.2f}')
        
        if highlight_min:
            min_idx = np.argmin(values)
            ax.scatter(angles[min_idx], values[min_idx], color='red', 
                      s=150, marker='v', zorder=5, label=f'Min: {values[min_idx]:.2f}')
        
        # Customize chart
        ax.set_ylim(0, 1)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Customize grid
        ax.grid(True, alpha=0.3)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        
        # Add legend if any special features are shown
        if show_average or highlight_max or highlight_min:
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        return fig
    
    def set_color_palette(self, palette_name: str):
        """
        Set the color palette for the charts.
        
        Parameters
        ----------
        palette_name : str
            Name of the color palette to use
        """
        if palette_name not in self.color_palettes:
            available = list(self.color_palettes.keys())
            raise ValueError(f"Palette '{palette_name}' not available. "
                           f"Available palettes: {available}")
        
        self.current_palette = self.color_palettes[palette_name]
        print(f"Color palette set to: {palette_name}")
    
    def get_available_palettes(self) -> List[str]:
        """
        Get list of available color palettes.
        
        Returns
        -------
        List[str]
            List of available palette names
        """
        return list(self.color_palettes.keys()) 