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

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


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


def get_kmer_freq(
    sequences: List[str], k: int
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int], Dict[str, float]]:
    """
    Calculate k-mer frequency statistics for a list of sequences.
    Args:
        sequences: List of peptide sequences.
        k: Length of k-mers to analyze.
    Returns:
        Tuple containing:
        - freq: Dictionary mapping kmer to its frequency (count/total_sequences)
        - counter: Dictionary mapping kmer to its total count across all sequences
        - seq_count: Dictionary mapping kmer to number of sequences containing it
        - seq_ratio: Dictionary mapping kmer to ratio of sequences containing it
    """
    counter = Counter()
    seq_counter = defaultdict(set)
    total = 0
    num_sequences = len(sequences)

    for idx, seq in enumerate(sequences):
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            counter[kmer] += 1
            seq_counter[kmer].add(idx)
            total += 1

    freq = {kmer: v / num_sequences for kmer, v in counter.items()}

    seq_count = {kmer: len(seq_counter[kmer]) for kmer in seq_counter}

    seq_ratio = {kmer: count / num_sequences for kmer, count in seq_count.items()}

    return freq, counter, seq_count, seq_ratio
