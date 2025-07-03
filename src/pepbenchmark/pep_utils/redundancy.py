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
import tempfile
from typing import List

import pandas as pd

from pepbenchmark.pep_utils.mmseq2 import (
    run_mmseqs_clustering,
    save_fasta,
)
from pepbenchmark.utils.logging import get_logger

logger = get_logger()


class RedundancyFilter:
    def __init__(self, identity: float):
        self.identity = identity

    def __call__(self, fasta: pd.Series, label: pd.Series) -> List[int]:
        assert len(fasta) == len(label), "Fasta and label must be of same length"
        self.origin_length = len(fasta)
        df = pd.concat([fasta, label], axis=1)
        df.columns = ["sequence", "label"]
        pos_df = df[df["label"] == 1]
        neg_df = df[df["label"] == 0]

        with tempfile.TemporaryDirectory() as tmp_root:
            pos_remain = self._dedup_and_get_index(pos_df, "positive", tmp_root)
            neg_remain = self._dedup_and_get_index(neg_df, "negative", tmp_root)

        remain_index = sorted(pos_remain + neg_remain)
        self.remain_index = remain_index
        self.remain_length = len(remain_index)
        self._calculate()
        return self.remain_index

    def _dedup_and_get_index(
        self, df: pd.DataFrame, label: str, root_dir: str
    ) -> List[int]:
        tmp_dir = os.path.join(root_dir, f"{label}_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        fasta_path = os.path.join(tmp_dir, f"{label}.fasta")
        cluster_dir = os.path.join(tmp_dir, "mmseqs")
        os.makedirs(cluster_dir, exist_ok=True)

        save_fasta(df["sequence"], fasta_path)

        cluster_tsv = run_mmseqs_clustering(
            fasta_path, cluster_dir, tmp_dir, self.identity
        )

        cluster_df = pd.read_csv(
            cluster_tsv, sep="\t", header=None, names=["rep_id", "member_id"]
        )
        rep_ids = cluster_df["rep_id"].unique().tolist()
        remain_index = [int(rid.replace("seq", "")) for rid in rep_ids]
        return remain_index

    def _calculate(self) -> None:
        remain_ratio = self.remain_length / self.origin_length
        logger.info(
            f"Redundancy filter: {remain_ratio:.2f} of the original data remains"
        )
