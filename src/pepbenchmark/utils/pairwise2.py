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

import random

import numpy as np
import pandas as pd
from Bio import pairwise2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm


def write_fasta(sequences: list[str], ids: list[str], filename: str) -> None:
    with open(filename, "w") as file:
        for sequence, id in zip(sequences, ids):
            file.write(f">{id}\n{sequence}\n")


def sequence_similarity_pairwise2(df, field_name="sequence"):
    sequences = df[field_name].values.tolist()
    n = len(sequences)
    sim_matrix = [[0.0] * n for _ in range(n)]

    for i in tqdm(range(n), desc="Computing similarities"):
        for j in range(i + 1, n):
            aln = pairwise2.align.globalxx(sequences[i], sequences[j], score_only=True)
            norm_score = aln / max(len(sequences[i]), len(sequences[j]))
            sim_matrix[i][j] = norm_score
            sim_matrix[j][i] = norm_score
    return sim_matrix


def _connected_components_clustering(sim_df, threshold=0.3):
    filtered_sim_df = sim_df.copy()
    filtered_sim_df[filtered_sim_df < threshold] = 0.0
    sparse_matrix = csr_matrix(filtered_sim_df)
    n_components, labels = connected_components(
        sparse_matrix, directed=False, return_labels=True
    )
    cluster_df = [{"cluster": labels[i], "member": i} for i in range(labels.shape[0])]
    return pd.DataFrame(cluster_df)


def split_by_similarity(
    df,
    sim_df,
    test_size: float = 0.2,
    valid_size: float = 0.0,
    seed: int = 0,
    threshold: float = 0.3,
):
    cluster_df = _connected_components_clustering(sim_df, threshold=threshold)
    partition_labs = cluster_df.cluster.to_numpy()
    unique_parts, part_counts = np.unique(partition_labs, return_counts=True)
    random.seed(seed)
    random.shuffle(unique_parts)

    df_size = df.shape[0]
    expected_test = test_size * df_size
    expected_valid = valid_size * df_size

    # Initialize empty lists for train, test, and valid sets
    test = []
    valid = []
    train = []

    # Precompute indices for test and valid partitions
    for part in unique_parts:
        part_indices = np.where(partition_labs == part)[0]

        if (len(test) + len(part_indices)) / df_size <= test_size:
            test.extend(part_indices)

    # Avoid test data points in valid set
    for part in unique_parts:
        part_indices = np.where(partition_labs == part)[0]
        remaining_indices = [i for i in part_indices if i not in test]

        if remaining_indices:
            if (len(valid) + len(remaining_indices)) / df_size <= valid_size:
                valid.extend(remaining_indices)
            else:
                train.extend(remaining_indices)

    # Warnings if the sizes of partitions are smaller than expected
    if len(test) < expected_test * 0.9:
        print(
            f"Warning: Proportion of test partition is smaller than expected: {(len(test) / df_size) * 100:.2f} %"
        )
    if len(valid) < expected_valid * 0.9:
        print(
            f"Warning: Proportion of validation partition is smaller than expected: {(len(valid) / df_size) * 100:.2f} %"
        )

    return train, test, valid
