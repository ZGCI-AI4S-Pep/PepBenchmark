# -*- coding: utf-8 -*-
"""Visualization tools for redundancy analysis.

This module provides comprehensive visualization capabilities for redundancy analysis results,
including similarity distributions, heatmaps, top-k analysis, and summary reports.

License: Apache-2.0
"""
from __future__ import annotations
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None

from pepbenchmark.redundancy.schemas import RedundancyReport


@dataclass 
class VisualizationConfig:
    """Visualization configuration"""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = "whitegrid"
    palette: str = "viridis"
    save_plots: bool = False
    output_dir: str = "./redundancy_plots"


def _upper_vals(mat: np.ndarray) -> np.ndarray:
    """Extract upper triangular values (excluding diagonal)"""
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]


class RedundancyVisualizer:
    """Redundancy analysis visualization toolkit"""
    
    def __init__(self, config: VisualizationConfig = None):
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("Visualization features require matplotlib and seaborn installation")
        self.config = config or VisualizationConfig()
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
        
    def plot_similarity_distribution(self, sim_matrix: np.ndarray, title: str = "Similarity Distribution"):
        """Plot similarity distribution histogram"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figsize)
        
        # Upper triangular similarity distribution
        upper_vals = _upper_vals(sim_matrix)
        ax1.hist(upper_vals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Similarity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Pairwise Similarity Distribution')
        ax1.axvline(upper_vals.mean(), color='red', linestyle='--', label=f'Mean: {upper_vals.mean():.3f}')
        ax1.legend()
        
        # Cumulative distribution
        sorted_vals = np.sort(upper_vals)
        cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax2.plot(sorted_vals, cumulative, 'b-', linewidth=2)
        ax2.set_xlabel('Similarity Threshold')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Similarity Cumulative Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Mark key thresholds
        for thresh in [0.7, 0.8, 0.9, 0.95]:
            idx = np.searchsorted(sorted_vals, thresh)
            if idx < len(cumulative):
                ax2.axvline(thresh, color='red', alpha=0.5, linestyle='--')
                ax2.text(thresh, cumulative[idx], f'{thresh}', rotation=90, 
                        verticalalignment='bottom', fontsize=8)
        
        plt.tight_layout()
        if self.config.save_plots:
            self._save_plot(fig, "similarity_distribution")
        return fig
    
    def plot_redundancy_heatmap(self, sim_matrix: np.ndarray, max_display: int = 100):
        """Plot similarity heatmap"""
        n = sim_matrix.shape[0]
        if n > max_display:
            # Random sampling for display
            idx = np.random.choice(n, max_display, replace=False)
            sim_sub = sim_matrix[np.ix_(idx, idx)]
            title = f"Similarity Heatmap (Random Sample {max_display}/{n})"
        else:
            sim_sub = sim_matrix
            title = f"Similarity Heatmap (All {n} Sequences)"
            
        fig, ax = plt.subplots(figsize=self.config.figsize)
        im = ax.imshow(sim_sub, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel('Sequence Index')
        ax.set_ylabel('Sequence Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similarity', rotation=270, labelpad=20)
        
        plt.tight_layout()
        if self.config.save_plots:
            self._save_plot(fig, "redundancy_heatmap")
        return fig
    
    def plot_topk_analysis(self, sim_matrix: np.ndarray, k_values: List[int] = [5, 10, 20, 50]):
        """Plot Top-K neighborhood analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figsize)
        
        n = sim_matrix.shape[0]
        k_data = {}
        
        # Calculate statistics for different K values
        for k in k_values:
            if k >= n: 
                continue
            topk_sims = []
            for i in range(n):
                row = sim_matrix[i].copy()
                row[i] = -1.0  # Exclude self
                top_indices = np.argpartition(-row, min(k, n-1))[:k]
                topk_sims.append(row[top_indices].mean())
            k_data[k] = np.array(topk_sims)
        
        if not k_data:
            return fig
            
        # 1. Top-K average similarity distribution
        for k, values in k_data.items():
            ax1.hist(values, bins=30, alpha=0.6, label=f'Top-{k}', density=True)
        ax1.set_xlabel('Average Similarity')
        ax1.set_ylabel('Density')
        ax1.set_title('Top-K Neighborhood Similarity Distribution')
        ax1.legend()
        
        # 2. Statistics comparison across different K values
        k_list = list(k_data.keys())
        means = [k_data[k].mean() for k in k_list]
        stds = [k_data[k].std() for k in k_list]
        
        ax2.errorbar(k_list, means, yerr=stds, marker='o', capsize=5)
        ax2.set_xlabel('K Value')
        ax2.set_ylabel('Average Similarity ± Std Dev')
        ax2.set_title('Top-K Statistics vs K Value')
        ax2.grid(True, alpha=0.3)
        
        # 3. Individual sequence Top-K similarity trajectories
        sample_indices = np.random.choice(n, min(10, n), replace=False)
        for idx in sample_indices:
            k_sims = []
            for k in k_list:
                row = sim_matrix[idx].copy()
                row[idx] = -1.0
                top_indices = np.argpartition(-row, min(k, n-1))[:k]
                k_sims.append(row[top_indices].mean())
            ax3.plot(k_list, k_sims, 'o-', alpha=0.7, linewidth=1)
        ax3.set_xlabel('K Value')
        ax3.set_ylabel('Top-K Average Similarity')
        ax3.set_title('Individual Sequence Top-K Trajectories')
        ax3.grid(True, alpha=0.3)
        
        # 4. Neighborhood size distribution under similarity thresholds
        thresholds = [0.7, 0.8, 0.9, 0.95]
        for thresh in thresholds:
            neighbor_counts = []
            for i in range(n):
                count = np.sum(sim_matrix[i] >= thresh) - 1  # Exclude self
                neighbor_counts.append(count)
            ax4.hist(neighbor_counts, bins=min(20, max(neighbor_counts)+1), 
                    alpha=0.6, label=f'Threshold≥{thresh}', density=True)
        ax4.set_xlabel('Neighborhood Size')
        ax4.set_ylabel('Density')
        ax4.set_title('Neighborhood Size Distribution by Threshold')
        ax4.legend()
        
        plt.tight_layout()
        if self.config.save_plots:
            self._save_plot(fig, "topk_analysis")
        return fig
    
    def plot_redundancy_summary(self, report: RedundancyReport):
        """Plot redundancy analysis summary"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figsize)

        # 1. Basic statistics radar chart
        categories = ['Uniqueness', 'Length Diversity', 'AA Diversity', 'Similarity Dispersion']
        values = [
            1.0 - report.basic.exact_duplicate_rate,
            1.0 / (1.0 + report.length.len_cv),
            report.aa.aa_entropy / 3.0,
            report.similarity.pair_std,
        ]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_closed = values + values[:1]
        angles_closed = angles + angles[:1]

        ax1.plot(angles_closed, values_closed, 'o-', linewidth=2, color='blue')
        ax1.fill(angles_closed, values_closed, alpha=0.25, color='blue')
        ax1.set_xticks(angles)
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 1)
        ax1.set_title('Data Quality Radar Chart')
        ax1.grid(True)

        # 2. Redundancy rate bar chart
        rates = report.redundancy.redundancy_rates
        thresholds = sorted(rates.keys())
        redundancy_rates = [rates[t] for t in thresholds]

        bars = ax2.bar(thresholds, redundancy_rates, color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Similarity Threshold')
        ax2.set_ylabel('Redundancy Rate')
        ax2.set_title('Redundancy Rate by Threshold')
        for bar, rate in zip(bars, redundancy_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{rate:.3f}', ha='center', va='bottom')

        # 3. Neff changes
        neffs = report.redundancy.neff
        neff_values = [neffs[t] for t in thresholds]
        ax3.plot(thresholds, neff_values, 'o-', linewidth=2, color='green')
        ax3.axhline(y=report.basic.n_total, color='red', linestyle='--',
                    label=f'Total: {report.basic.n_total}')
        ax3.set_xlabel('Similarity Threshold')
        ax3.set_ylabel('Effective Sequence Number (Neff)')
        ax3.set_title('Neff vs Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Data quality summary text
        rec = report.recommendation
        ax4.text(0.1, 0.8, f"Recommended Threshold: {rec.recommended_threshold:.2f}",
                 fontsize=14, weight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f"Data Quality Score: {rec.data_quality_score:.3f}",
                 fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.4, f"Redundancy Severity: {rec.redundancy_severity}",
                 fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.2, f"Total Sequences: {report.basic.n_total}",
                 fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.1, f"Unique Sequences: {report.basic.n_unique}",
                 fontsize=10, transform=ax4.transAxes)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('Analysis Summary')
        ax4.axis('off')

        plt.tight_layout()
        if self.config.save_plots:
            self._save_plot(fig, "redundancy_summary")
        return fig
    
    def _save_plot(self, fig, name: str):
        """Save plot to file"""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        fig.savefig(f"{self.config.output_dir}/{name}.png", 
                   dpi=self.config.dpi, bbox_inches='tight')
        
    def create_comprehensive_report(self, report: RedundancyReport):
        """Create comprehensive visualization report"""
        figures = []
        
        if report.sim_matrix is not None:
            figures.append(self.plot_similarity_distribution(report.sim_matrix))
            figures.append(self.plot_redundancy_heatmap(report.sim_matrix))
            figures.append(self.plot_topk_analysis(report.sim_matrix))
        
        figures.append(self.plot_redundancy_summary(report))
        return figures
