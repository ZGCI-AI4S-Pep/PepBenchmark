# Copyright ZGCA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pepbenchmark.analyze.properties import calculate_properties
from pepbenchmark.analyze.statistic import compute_aa_freq, get_kmer_freq


def visualize_property_distribution_single(
    seqs: list[str],
    properties: Optional[list[str]] = None,
    plot_type: str = "kde",
    bins: int = 20,
    logger=None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize property distributions for a single group of sequences.
    Args:
        seqs: List of peptide sequences.
        properties: List of property names to visualize. If None, auto-detect.
        plot_type: 'hist' for histogram, 'kde' for density plot.
        bins: Number of bins for histogram.
        logger: Optional logger for info output.
    """
    df = calculate_properties(seqs, properties)
    if properties is None:
        properties = [
            col
            for col in df.columns
            if col != "sequence" and pd.api.types.is_numeric_dtype(df[col])
        ]
    n_props = len(properties)
    n_cols = 3
    n_rows = math.ceil(n_props / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, prop in enumerate(properties):
        ax = axes[i]
        data = df[prop].dropna().to_numpy()
        if plot_type == "hist":
            sns.histplot(
                data,
                bins=bins,
                color="blue",
                kde=False,
                stat="density",
                element="step",
                fill=False,
                ax=ax,
            )
        elif plot_type == "kde":
            sns.kdeplot(data, color="blue", ax=ax)
        else:
            raise ValueError(
                f"Unsupported plot_type: {plot_type}. Choose 'hist' or 'kde'."
            )
        ax.set_title(f"{prop} distribution")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def visualize_property_distribution_compare(
    seqs1: list[str],
    seqs2: list[str],
    properties: Optional[list[str]] = None,
    plot_type: str = "kde",
    bins: int = 20,
    label1: str = "Group 1",
    label2: str = "Group 2",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize property distributions between two groups of sequences for comparison.
    Args:
        seqs1: List of peptide sequences (group 1).
        seqs2: List of peptide sequences (group 2).
        properties: List of property names to visualize. If None, auto-detect.
        plot_type: 'hist' for histogram, 'kde' for density plot.
        bins: Number of bins for histogram.
        label1: Custom label for group 1.
        label2: Custom label for group 2.
        save_path: Optional path to save the plot.
    """
    df1 = calculate_properties(seqs1, properties)
    df2 = calculate_properties(seqs2, properties)
    if properties is None:
        properties = [
            col
            for col in df1.columns
            if col != "sequence" and pd.api.types.is_numeric_dtype(df1[col])
        ]
    n_props = len(properties)
    n_cols = 3
    n_rows = math.ceil(n_props / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, prop in enumerate(properties):
        ax = axes[i]
        data1 = df1[prop].dropna().to_numpy()
        data2 = df2[prop].dropna().to_numpy()
        if plot_type == "hist":
            sns.histplot(
                data1,
                bins=bins,
                color="blue",
                label=label1,
                kde=False,
                stat="density",
                element="step",
                fill=False,
                ax=ax,
            )
            sns.histplot(
                data2,
                bins=bins,
                color="red",
                label=label2,
                kde=False,
                stat="density",
                element="step",
                fill=False,
                ax=ax,
            )
        elif plot_type == "kde":
            sns.kdeplot(data1, color="blue", label=label1, ax=ax)
            sns.kdeplot(data2, color="red", label=label2, ax=ax)
        else:
            raise ValueError(
                f"Unsupported plot_type: {plot_type}. Choose 'hist' or 'kde'."
            )
        ax.set_title(f"{prop} distribution")
        ax.legend()
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def visualize_aas_distribution_single(
    seqs: list[str], save_path: Optional[str] = None
) -> None:
    """
    Visualize the amino acid frequency distribution for a single group of sequences.
    Args:
        seqs: List of peptide sequences.
    """
    freq_df = compute_aa_freq(seqs)
    plt.figure(figsize=(10, 5))
    sns.barplot(x="AA", y="freq", data=freq_df, color="blue")
    plt.title("Amino Acid Frequency Distribution")
    plt.ylabel("Frequency")
    plt.xlabel("Amino Acid")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def visualize_aas_distribution_compare(
    seqs1: list[str],
    seqs2: list[str],
    label1: str = "Group 1",
    label2: str = "Group 2",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize and compare the amino acid frequency distributions between two groups of sequences.
    Args:
        seqs1: List of peptide sequences (group 1).
        seqs2: List of peptide sequences (group 2).
        label1: Custom label for group 1.
        label2: Custom label for group 2.
        save_path: Optional path to save the plot.
    """
    freq_df1 = compute_aa_freq(seqs1)
    freq_df2 = compute_aa_freq(seqs2)
    freq_df1 = freq_df1.rename(columns={"freq": "freq1"})
    freq_df2 = freq_df2.rename(columns={"freq": "freq2"})
    merged = pd.merge(freq_df1, freq_df2, on="AA", how="outer").fillna(0)
    x = merged["AA"]
    width = 0.35
    plt.figure(figsize=(12, 5))
    plt.bar(x, merged["freq1"], width=width, label=label1, color="blue", align="center")
    plt.bar(x, merged["freq2"], width=width, label=label2, color="red", align="edge")
    plt.title("Amino Acid Frequency Distribution Comparison")
    plt.ylabel("Frequency")
    plt.xlabel("Amino Acid")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_similarity_distribution(
    similarities: List[float], threshold: float = 0.9, save_path: Optional[str] = None
):
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=20, alpha=0.5, color="skyblue", edgecolor="black")
    above_threshold = sum(sim > threshold for sim in similarities)
    ratio = above_threshold / len(similarities)
    plt.axvline(
        threshold, color="red", linestyle="--", label=f"Threshold = {threshold}"
    )
    plt.text(
        threshold + 0.01, plt.ylim()[1] * 0.8, f">{threshold}: {ratio:.1%}", color="red"
    )

    # Fix x-axis ticks to show proper similarity values
    min_sim = min(similarities)
    max_sim = max(similarities)
    # Create reasonable tick spacing based on the range
    if max_sim - min_sim <= 0.1:
        tick_spacing = 0.02
    elif max_sim - min_sim <= 0.5:
        tick_spacing = 0.05
    else:
        tick_spacing = 0.1

    # Generate ticks with proper spacing
    ticks = np.arange(
        np.floor(min_sim * 100) / 100,
        np.ceil(max_sim * 100) / 100 + tick_spacing,
        tick_spacing,
    )
    plt.xticks(ticks)

    plt.xlabel("Max Sequence Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Similarity Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_similarity_distribution_comparison(
    similarity_scores_1: List[float],
    similarity_scores_2: List[float],
    label_1: str = "Before Deduplication",
    label_2: str = "After Deduplication",
    threshold: float = 0.9,
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(10, 6))
    plt.hist(
        similarity_scores_1,
        bins=20,
        label=label_1,
        alpha=0.5,
        color="blue",
        edgecolor="black",
    )
    plt.hist(
        similarity_scores_2,
        bins=20,
        label=label_2,
        alpha=0.5,
        color="green",
        edgecolor="black",
    )
    plt.axvline(
        threshold, color="red", linestyle="--", label=f"Threshold = {threshold}"
    )
    above_1 = sum(sim > threshold for sim in similarity_scores_1)
    ratio_1 = above_1 / len(similarity_scores_1)
    plt.text(
        threshold + 0.01,
        plt.ylim()[1] * 0.85,
        f"{label_1}: {ratio_1:.1%}",
        color="blue",
    )
    above_2 = sum(sim > threshold for sim in similarity_scores_2)
    ratio_2 = above_2 / len(similarity_scores_2)
    plt.text(
        threshold + 0.01,
        plt.ylim()[1] * 0.75,
        f"{label_2}: {ratio_2:.1%}",
        color="green",
    )

    # Fix x-axis ticks to show proper similarity values
    all_similarities = similarity_scores_1 + similarity_scores_2
    min_sim = min(all_similarities)
    max_sim = max(all_similarities)
    # Create reasonable tick spacing based on the range
    if max_sim - min_sim <= 0.1:
        tick_spacing = 0.02
    elif max_sim - min_sim <= 0.5:
        tick_spacing = 0.05
    else:
        tick_spacing = 0.1

    # Generate ticks with proper spacing
    ticks = np.arange(
        np.floor(min_sim * 100) / 100,
        np.ceil(max_sim * 100) / 100 + tick_spacing,
        tick_spacing,
    )
    plt.xticks(ticks)

    plt.xlabel("Max Sequence Similarity")
    plt.ylabel("Frequency")
    plt.title("Similarity Distribution Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def visualize_top_kmer_frequency(
    sequences: List[str], k: int = 6, top_n: int = 20, save_path: Optional[str] = None
) -> None:
    """
    Visualize the frequency of top N k-mers based on sequence count.
    Args:
        sequences: List of peptide sequences.
        k: Length of k-mers to analyze.
        top_n: Number of top k-mers to display (default: 20).
        save_path: Optional path to save the plot.
        min_frequency: Minimum frequency threshold to filter kmers (default: 0.001).
    """
    if len(sequences) == 0:
        print("Warning: No sequences provided for analysis.")
        return

    _, _, seq_count, seq_ratio = get_kmer_freq(sequences, k)

    # Sort by sequence count and get top N
    sorted_kmers = sorted(seq_count.items(), key=lambda x: x[1], reverse=True)[:top_n]

    kmers = [item[0] for item in sorted_kmers]
    frequencies = [seq_ratio[item[0]] for item in sorted_kmers]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        range(len(kmers)), frequencies, color="skyblue", edgecolor="navy", alpha=0.7
    )

    # Add value labels on bars
    total_sequences = len(sequences)
    for _, (bar, freq) in enumerate(zip(bars, frequencies)):
        # Calculate position for text label
        x_pos = bar.get_x() + bar.get_width() / 2
        y_pos = bar.get_height() + max(frequencies) * 0.02  # Dynamic offset

        # Calculate actual sequence count
        seq_count = int(freq * total_sequences)

        # Add text with fraction format
        plt.text(
            x_pos,
            y_pos,
            f"{seq_count}/{total_sequences}",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
        )

    # Adjust y-axis limits to accommodate labels
    max_freq = max(frequencies) if frequencies else 0.001
    plt.ylim(0, max_freq * 1.2)  # Add 20% space for labels

    plt.xlabel("K-mer")
    plt.ylabel("Sequences containing k-mer / Total sequences")
    plt.title(f"Top {top_n} {k}-mer Frequencies by Sequence Count")
    plt.xticks(range(len(kmers)), kmers, rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Add summary statistics
    if len(sequences) > 0:
        avg_freq = sum(frequencies) / len(frequencies)
        plt.figtext(
            0.02,
            0.02,
            f"Average frequency: {avg_freq:.6f}",
            fontsize=10,
            style="italic",
        )

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def compare_kmer_frequency(
    sequences1: List[str],
    sequences2: List[str],
    label1: str = "Group 1",
    label2: str = "Group 2",
    k: int = 3,
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Compare k-mer frequencies between two groups of sequences.
    Args:
        sequences1: First group of peptide sequences.
        sequences2: Second group of peptide sequences.
        k: Length of k-mers to analyze.
        top_n: Number of top k-mers to display (default: 20).
        save_path: Optional path to save the plot.
    """
    if len(sequences1) == 0 or len(sequences2) == 0:
        print("Warning: One or both sequence groups are empty.")
        return

    _, _, seq_count1, seq_ratio1 = get_kmer_freq(sequences1, k)
    _, _, seq_count2, seq_ratio2 = get_kmer_freq(sequences2, k)

    # Get all unique kmers from both groups
    all_kmers = set(seq_count1.keys()) | set(seq_count2.keys())

    # Calculate combined score for ranking (sum of sequence counts from both groups)
    combined_scores = {}
    for kmer in all_kmers:
        score1 = seq_count1.get(kmer, 0)
        score2 = seq_count2.get(kmer, 0)
        combined_scores[kmer] = score1 + score2

    # Sort by combined score and get top N
    sorted_kmers = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[
        :top_n
    ]

    kmers = [item[0] for item in sorted_kmers]
    freq1 = [seq_ratio1.get(kmer, 0) for kmer in kmers]
    freq2 = [seq_ratio2.get(kmer, 0) for kmer in kmers]

    x = np.arange(len(kmers))
    width = 0.35

    plt.figure(figsize=(14, 8))
    bars1 = plt.bar(
        x - width / 2, freq1, width, label=label1, color="skyblue", alpha=0.7
    )
    bars2 = plt.bar(
        x + width / 2, freq2, width, label=label2, color="lightcoral", alpha=0.7
    )

    # Add value labels on bars
    total_sequences1 = len(sequences1)
    total_sequences2 = len(sequences2)
    for _, (bar1, bar2, f1, f2) in enumerate(zip(bars1, bars2, freq1, freq2)):
        if f1 > 0:
            seq_count1 = int(f1 * total_sequences1)
            plt.text(
                bar1.get_x() + bar1.get_width() / 2,
                bar1.get_height() + 0.003,
                f"{seq_count1}/{total_sequences1}",
                ha="center",
                va="bottom",
                fontsize=7,
                bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "alpha": 0.8},
            )
        if f2 > 0:
            seq_count2 = int(f2 * total_sequences2)
            plt.text(
                bar2.get_x() + bar2.get_width() / 2,
                bar2.get_height() + 0.003,
                f"{seq_count2}/{total_sequences2}",
                ha="center",
                va="bottom",
                fontsize=7,
                bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "alpha": 0.8},
            )

    # Adjust y-axis limits to accommodate labels
    max_freq = max(max(freq1) if freq1 else 0, max(freq2) if freq2 else 0)
    plt.ylim(0, max_freq * 1.2)  # Add 20% space for labels

    plt.xlabel("K-mer")
    plt.ylabel("Sequences containing k-mer / Total sequences")
    plt.title(
        f"Top {len(kmers)} {k}-mer Frequencies Comparison\n({label1}: {total_sequences1}, {label2}: {total_sequences2})"
    )
    plt.xticks(x, kmers, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Add summary statistics
    if len(sequences1) > 0 and len(sequences2) > 0:
        avg_freq1 = sum(freq1) / len(freq1) if freq1 else 0
        avg_freq2 = sum(freq2) / len(freq2) if freq2 else 0
        plt.figtext(
            0.02,
            0.02,
            f"Avg freq: {label1}={avg_freq1:.6f}, {label2}={avg_freq2:.6f}",
            fontsize=10,
            style="italic",
        )

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def visualize_regression_labels(
    sequences: List[str],
    labels: List[float],
    plot_type: str = "hist",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize the distribution of regression task labels.

    Args:
        sequences: List of peptide sequences.
        labels: List of corresponding label values (float).
        plot_type: Type of plot - 'hist' for histogram, 'violin' for violin plot.
        save_path: Optional path to save the plot.
    """
    if len(sequences) != len(labels):
        raise ValueError("Sequences and labels must have the same length.")

    if len(sequences) == 0:
        print("Warning: No data provided for visualization.")
        return

    # Convert to numpy array for easier manipulation
    labels_array = np.array(labels)

    # Calculate basic statistics
    mean_val = np.mean(labels_array)
    median_val = np.median(labels_array)
    std_val = np.std(labels_array)
    min_val = np.min(labels_array)
    max_val = np.max(labels_array)

    # Detect outliers using IQR method
    outliers = []
    q1 = np.percentile(labels_array, 25)
    q3 = np.percentile(labels_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = labels_array[(labels_array < lower_bound) | (labels_array > upper_bound)]

    # Create the plot
    plt.figure(figsize=(10, 6))

    if plot_type == "hist":
        # Histogram with KDE
        plt.hist(
            labels_array,
            bins=30,
            alpha=0.7,
            color="skyblue",
            edgecolor="navy",
            density=True,
            label="Histogram",
        )

        # Add KDE curve
        from scipy.stats import gaussian_kde

        kde_estimator = gaussian_kde(labels_array)
        x_range = np.linspace(min_val, max_val, 200)
        kde_curve = kde_estimator(x_range)
        plt.plot(x_range, kde_curve, color="red", linewidth=2, label="KDE", alpha=0.8)
        plt.legend()

        plt.xlabel("Label Values")
        plt.ylabel("Density")

    elif plot_type == "violin":
        # Violin plot
        violin_parts = plt.violinplot(labels_array, showmeans=True, showmedians=True)
        bodies = violin_parts.get("bodies", [])
        if isinstance(bodies, list) and len(bodies) > 0:
            bodies[0].set_facecolor("skyblue")
            bodies[0].set_alpha(0.7)
        if "cmeans" in violin_parts:
            violin_parts["cmeans"].set_color("red")
        if "cmedians" in violin_parts:
            violin_parts["cmedians"].set_color("blue")

        plt.ylabel("Label Values")

    else:
        raise ValueError(
            f"Unsupported plot_type: {plot_type}. Choose 'hist', 'box', or 'violin'."
        )

    # Add title and grid
    plt.title("Regression Labels Distribution")
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}\nN: {len(labels_array)}"
    if len(outliers) > 0:
        stats_text += f"\nOutliers: {len(outliers)}"

    plt.figtext(
        0.02,
        0.02,
        stats_text,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8},
        verticalalignment="bottom",
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
