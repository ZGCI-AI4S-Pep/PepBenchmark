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
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from tqdm import tqdm


def calculate_properties(
    sequences: list[str], properties: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Calculate selected peptide properties.
    Args:
        sequences: List of peptide sequences.
        properties: List of property names to calculate. If None, calculate all supported properties.
    Returns:
        DataFrame containing selected properties and corresponding bins.
    Raises:
        ValueError: If any property in properties is not supported.
    """
    all_props = {
        "mw": lambda analysis: analysis.molecular_weight(),
        "hydrophobicity": lambda analysis: analysis.gravy(),
        "charge": lambda analysis: analysis.charge_at_pH(7.0),
        "isoelectricpoint": lambda analysis: analysis.isoelectric_point(),
    }
    if properties is None:
        selected_props = all_props
    else:
        invalid_props = [k for k in properties if k not in all_props]
        if invalid_props:
            raise ValueError(
                f"The following properties are not supported: {invalid_props}"
            )
        selected_props = {k: v for k, v in all_props.items() if k in properties}
    df = pd.DataFrame({"sequence": sequences})
    df["length"] = df["sequence"].apply(len)

    def _calc_props(seq):
        analysis = ProteinAnalysis(seq)
        props = {k: func(analysis) for k, func in selected_props.items()}
        return pd.Series(props)

    df = pd.concat([df, df["sequence"].apply(_calc_props)], axis=1)
    return df


def compute_aa_freq(seqs: List[str]) -> pd.DataFrame:
    """
    Calculate the frequency of each amino acid in a list of sequences.
    Args:
        seqs: List of peptide sequences (strings).
    Returns:
        DataFrame containing each amino acid and its frequency.
    """
    aa_order = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    freq_counter = Counter()
    total_count = 0
    for seq in seqs:
        freq_counter.update(seq)
        total_count += len(seq)
    freq_list = []
    for aa in aa_order:
        freq = freq_counter[aa] / total_count if total_count > 0 else 0
        freq_list.append({"AA": aa, "freq": freq})
    return pd.DataFrame(freq_list)


def compare_properties_distribution(
    seqs1: list[str], seqs2: list[str], properties: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Compare the distribution of given properties between two sets of sequences.
    Calculates metrics such as mean difference, Jensen-Shannon distance, KS statistic, and quantile differences for each property.
    Args:
        seqs1: List of sequences (group 1)
        seqs2: List of sequences (group 2)
        properties: List of property names to evaluate (e.g., ["mw", "hydrophobicity", ...]). If None, auto-detect numeric columns.
    Returns:
        DataFrame with metrics for each property.
    Raises:
        ValueError: If any property in properties is not present in the calculated property columns.
    """
    # Calculate properties for both groups
    df1 = calculate_properties(seqs1, properties)
    df2 = calculate_properties(seqs2, properties)
    if properties is None:
        properties = [
            col
            for col in df1.columns
            if col != "sequence" and pd.api.types.is_numeric_dtype(df1[col])
        ]
    else:
        # Validate properties
        invalid_props = [p for p in properties if p not in df1.columns]
        if invalid_props:
            raise ValueError(
                f"The following properties are not defined: {invalid_props}"
            )
    results = []
    for prop in properties:
        arr1 = df1[prop].dropna().to_numpy()
        arr2 = df2[prop].dropna().to_numpy()
        # Mean difference
        mean_diff = np.mean(arr1) - np.mean(arr2)
        # Jensen-Shannon distance (use histogram)
        hist1, bins = np.histogram(arr1, bins=30, density=True)
        hist2, _ = np.histogram(arr2, bins=bins, density=True)
        js_dist = jensenshannon(hist1, hist2)
        # KS statistic
        ks_stat, ks_p = ks_2samp(arr1, arr2)
        # Quantile differences
        quantiles = [0.25, 0.5, 0.75]
        quantile_diffs = [
            np.quantile(arr1, q) - np.quantile(arr2, q) for q in quantiles
        ]
        results.append(
            {
                "property": prop,
                "mean_diff": mean_diff,
                "js_distance": js_dist,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_p,
                "q25_diff": quantile_diffs[0],
                "q50_diff": quantile_diffs[1],
                "q75_diff": quantile_diffs[2],
            }
        )
    return pd.DataFrame(results)


def visualize_property_distribution_single(
    seqs: list[str],
    properties: Optional[list[str]] = None,
    plot_type: str = "kde",
    bins: int = 20,
    logger=None,
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
    plt.show()


def visualize_property_distribution_compare(
    seqs1: list[str],
    seqs2: list[str],
    properties: Optional[list[str]] = None,
    plot_type: str = "kde",
    bins: int = 20,
    logger=None,
) -> None:
    """
    Visualize property distributions between two groups of sequences for comparison.
    Args:
        seqs1: List of peptide sequences (group 1).
        seqs2: List of peptide sequences (group 2).
        properties: List of property names to visualize. If None, auto-detect.
        plot_type: 'hist' for histogram, 'kde' for density plot.
        bins: Number of bins for histogram.
        logger: Optional logger for info output.
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
                label="Group 1",
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
                label="Group 2",
                kde=False,
                stat="density",
                element="step",
                fill=False,
                ax=ax,
            )
        elif plot_type == "kde":
            sns.kdeplot(data1, color="blue", label="Group 1", ax=ax)
            sns.kdeplot(data2, color="red", label="Group 2", ax=ax)
        else:
            raise ValueError(
                f"Unsupported plot_type: {plot_type}. Choose 'hist' or 'kde'."
            )
        ax.set_title(f"{prop} distribution")
        ax.legend()
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig("property_distribution_compare.png")
    plt.show()


def visualize_aas_distribution_single(seqs: list[str]) -> None:
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
    plt.show()


def visualize_aas_distribution_compare(seqs1: list[str], seqs2: list[str]) -> None:
    """
    Visualize and compare the amino acid frequency distributions between two groups of sequences.
    Args:
        seqs1: List of peptide sequences (group 1).
        seqs2: List of peptide sequences (group 2).
    """
    freq_df1 = compute_aa_freq(seqs1)
    freq_df2 = compute_aa_freq(seqs2)
    freq_df1 = freq_df1.rename(columns={"freq": "freq1"})
    freq_df2 = freq_df2.rename(columns={"freq": "freq2"})
    merged = pd.merge(freq_df1, freq_df2, on="AA", how="outer").fillna(0)
    x = merged["AA"]
    width = 0.35
    plt.figure(figsize=(12, 5))
    plt.bar(
        x, merged["freq1"], width=width, label="Group 1", color="blue", align="center"
    )
    plt.bar(x, merged["freq2"], width=width, label="Group 2", color="red", align="edge")
    plt.title("Amino Acid Frequency Distribution Comparison")
    plt.ylabel("Frequency")
    plt.xlabel("Amino Acid")
    plt.legend()
    plt.tight_layout()
    plt.savefig("aas_distribution_compare.png")
    plt.show()


# =================== Redundancy/SIMILARITY ANALYSIS UTILS ===================


def sliding_window_aar(pair: Tuple[str, str]) -> float:
    seq1, seq2 = pair
    max_match = 0
    len1, len2 = len(seq1), len(seq2)
    for offset in range(-len2 + 1, len1):
        matches = 0
        for i in range(len1):
            j = i - offset
            if 0 <= j < len2:
                if seq1[i] == seq2[j]:
                    matches += 1
        max_match = max(max_match, matches)
    return max_match / max(len1, len2)


def _init_worker(_samples, _smi_func):
    global shared_samples, shared_smi_function
    shared_samples = _samples
    shared_smi_function = _smi_func


def _wrapper(pair: Tuple[int, int]) -> Tuple[Tuple[int, int], float | None]:
    i, j = pair
    try:
        return ((i, j), shared_smi_function((shared_samples[i], shared_samples[j])))
    except Exception:
        return ((i, j), None)


def calculate_similarity_each_pair(
    samples: List[str], smi_function=sliding_window_aar, processes: int = None
) -> List[Tuple[Tuple[int, int], float]]:
    n = len(samples)
    if processes is None:
        processes = max(1, cpu_count() - 1)
    index_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    with Pool(
        processes, initializer=_init_worker, initargs=(samples, smi_function)
    ) as pool:
        results = list(tqdm(pool.imap(_wrapper, index_pairs), total=len(index_pairs)))
    # 只返回sim不为None的pair
    return [(ij, sim) for (ij, sim) in results if sim is not None]


def get_max_similarity_per_sample(
    sequences: List[str], processes: int = None
) -> List[float]:
    # 保证processes为int类型
    n_cpu = max(1, cpu_count() - 1)
    proc = processes if processes is not None else n_cpu
    pair_similarities = calculate_similarity_each_pair(
        sequences, sliding_window_aar, proc
    )
    n = len(sequences)
    max_sim_per_seq = [0.0] * n
    for (i, j), sim in pair_similarities:
        max_sim_per_seq[i] = max(max_sim_per_seq[i], sim)
        max_sim_per_seq[j] = max(max_sim_per_seq[j], sim)
    return max_sim_per_seq


def plot_similarity_distribution(similarities: List[float], threshold: float = 0.9):
    plt.hist(similarities, bins=20, alpha=0.5)
    above_threshold = sum(sim > threshold for sim in similarities)
    ratio = above_threshold / len(similarities)
    plt.axvline(
        threshold, color="red", linestyle="--", label=f"Threshold = {threshold}"
    )
    plt.text(
        threshold + 0.01, plt.ylim()[1] * 0.8, f">{threshold}: {ratio:.1%}", color="red"
    )
    plt.xticks(np.arange(min(similarities), max(similarities) + 0.1, 0.1))
    plt.xlabel("Max Sequence Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Similarity Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("similarity_distribution.png")
    plt.show()


def plot_similarity_distribution_comparison(
    similarity_scores_1: List[float],
    similarity_scores_2: List[float],
    label_1: str = "Before Deduplication",
    label_2: str = "After Deduplication",
    threshold: float = 0.9,
):
    plt.hist(similarity_scores_1, bins=20, label=label_1, alpha=0.5, color="blue")
    plt.hist(similarity_scores_2, bins=20, label=label_2, alpha=0.5, color="green")
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
    plt.xticks(np.arange(min(similarity_scores_1), max(similarity_scores_1) + 0.1, 0.1))
    plt.xlabel("Max Sequence Similarity")
    plt.ylabel("Frequency")
    plt.title("Similarity Distribution Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("similarity_distribution_comparison.png")
    plt.show()
