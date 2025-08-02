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
from typing import Dict, List, Tuple

import pandas as pd


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
