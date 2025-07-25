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

import os
import random
import tempfile
from typing import List

from pepbenchmark.pep_utils.cdhit import (
    get_representative_seqs,
    run_cdhit_clustering,
)
from pepbenchmark.pep_utils.mmseq2 import (
    get_representative_ids_mmseq,
    run_mmseqs_clustering,
    save_fasta,
)
from pepbenchmark.utils.analyze import (
    calculate_similarity_each_pair,
    get_max_similarity_per_sample,
    plot_similarity_distribution,
    plot_similarity_distribution_comparison,
    sliding_window_aar,
)
from pepbenchmark.utils.logging import get_logger

logger = get_logger()


class Redundancy:
    """
    Class for redundancy analysis and filtering.
    """

    def __init__(self):
        self.threshold = None
        self.similarity_scores = None
        self._cached_data_hash = None

    def _get_or_calculate_similarity(
        self, sequences: List[str], threshold: float, processes: int = None
    ) -> List[float]:
        current_data_hash = self._get_data_hash(sequences)
        if (
            self.similarity_scores is not None
            and self.threshold == threshold
            and self._cached_data_hash == current_data_hash
        ):
            logger.info("Using cached similarity_scores")
            return self.similarity_scores
        logger.info("Computing max similarity distribution ...")
        self._cached_data_hash = current_data_hash
        self.threshold = threshold
        self.similarity_scores = get_max_similarity_per_sample(sequences, processes)
        return self.similarity_scores

    def _get_data_hash(self, data: List[str]) -> str:
        import hashlib

        data_str = "".join(data)
        return hashlib.md5(data_str.encode()).hexdigest()

    def analyze(
        self, sequences: List[str], threshold: float = 0.9, processes: int = 16
    ) -> None:
        logger.info("Computing max similarity distribution ...")
        max_sim = self._get_or_calculate_similarity(sequences, threshold, processes)
        self.similarity_scores = max_sim
        plot_similarity_distribution(max_sim, threshold)

    def deduplicate(
        self,
        sequences: List[str],
        threshold: float = 0.9,
        dedup_method: str = "aar",
        identity: float = 0.9,
        processes: int = 16,
        visualization: bool = True,
    ) -> List[str]:
        logger.info(f"Running deduplication using method: {dedup_method}...")
        remain_seqs = self._dedup_and_get_index(
            sequences, dedup_method, identity, processes=processes
        )
        logger.info(f"Redundancy filter: {len(remain_seqs)} / {len(sequences)} remains")
        if visualization:
            similarity_scores_before = self._get_or_calculate_similarity(
                sequences, threshold, processes
            )
            similarity_scores_after = get_max_similarity_per_sample(
                remain_seqs, processes
            )
            plot_similarity_distribution_comparison(
                similarity_scores_before, similarity_scores_after, threshold=threshold
            )
        return remain_seqs

    def _dedup_and_get_index(
        self,
        sequences: List[str],
        dedup_method: str,
        identity: float = 0.9,
        processes=None,
    ) -> List[str]:
        if dedup_method == "aar":
            return self._filter_by_seq_similarity(sequences, identity, processes)
        elif dedup_method in ["mmseqs", "cdhit"]:
            return self._filter_by_cluster(sequences, dedup_method, identity)
        else:
            raise ValueError(f"Unsupported method: {dedup_method}")

    def _filter_by_seq_similarity(
        self, sequences: List[str], threshold: float, processes=None
    ) -> List[str]:
        from multiprocessing import cpu_count

        n_cpu = max(1, cpu_count() - 1)
        proc = processes if processes is not None else n_cpu
        pair_similarity = calculate_similarity_each_pair(
            sequences, sliding_window_aar, proc
        )
        remaining_indices = set(range(len(sequences)))
        sorted_pairs = sorted(pair_similarity, key=lambda x: -x[1])
        for (i, j), sim in sorted_pairs:
            if sim < threshold:
                break
            if i in remaining_indices and j in remaining_indices:
                remove_idx = random.choice([i, j])
                remaining_indices.remove(remove_idx)
        remain_indices = list(remaining_indices)
        remain_seqs = [sequences[i] for i in remain_indices]
        return remain_seqs

    def _filter_by_cluster(
        self, sequences: List[str], dedup_method: str, threshold: float
    ) -> List[str]:
        with tempfile.TemporaryDirectory() as tmp_root:
            tmp_dir = os.path.join(tmp_root, "dedup_tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            fasta_path = os.path.join(tmp_dir, "input.fasta")
            save_fasta(sequences, fasta_path)
            if dedup_method == "mmseqs":
                cluster_dir = os.path.join(tmp_dir, "mmseqs_result")
                os.makedirs(cluster_dir, exist_ok=True)
                cluster_tsv = run_mmseqs_clustering(
                    fasta_path, cluster_dir, tmp_dir, threshold
                )
                rep_ids = get_representative_ids_mmseq(cluster_tsv)
                remain_indices = [int(rid.replace("seq", "")) for rid in rep_ids]
                remain_seqs = [sequences[i] for i in remain_indices]
            elif dedup_method == "cdhit":
                result_fasta_path = run_cdhit_clustering(
                    fasta_path, tmp_dir, threshold, tolerant=5
                )
                print(result_fasta_path)
                remain_seqs = get_representative_seqs(result_fasta_path)
        return remain_seqs


if __name__ == "__main__":
    from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager

    dataset_name = "bbp"  # Change this to your dataset name

    dataset_manager = SingleTaskDatasetManager(
        dataset_name=dataset_name, official_feature_names=["fasta"]
    )

    pos_seqs = dataset_manager.get_positive_sequences()

    rd = Redundancy()

    rd.analyze(pos_seqs)
    # remain_seqs = rd.deduplicate(pos_seqs, threshold=0.9, dedup_method="mmseqs", identity=0.9)
    # remain_seqs = rd.deduplicate(pos_seqs, threshold=0.8, dedup_method="needleman", identity=0.8,visualization=True)
    # df = pd.DataFrame(remain_seqs,columns=["sequence"])
    # df.to_csv("remain_seqs.csv",index=False)
    # remain_seqs = rd.deduplicate(pos_seqs, threshold=0.9, dedup_method="cdhit", identity=0.7, visualization=True)
    remain_seqs = rd.deduplicate(
        pos_seqs, threshold=0.9, dedup_method="mmseqs", identity=0.8
    )
