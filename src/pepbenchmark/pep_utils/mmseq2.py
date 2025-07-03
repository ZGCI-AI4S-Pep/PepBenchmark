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
import subprocess
from typing import Dict, List

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from pepbenchmark.utils.logging import get_logger

logger = get_logger()


def save_fasta(fasta: pd.Series, path: str) -> None:
    """
    Save a pandas Series of sequences to a FASTA file.

    Args:
        fasta: pandas.Series, where index is ID and value is sequence.
        path: Output FASTA file path.
    """
    records = [
        SeqRecord(Seq(seq), id=str(idx), description="") for idx, seq in fasta.items()
    ]
    SeqIO.write(records, path, "fasta")


def run_mmseqs_clustering(
    input_fasta: str, output_dir: str, tmp_dir: str, identity: float
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    db = os.path.join(output_dir, "db")
    result = os.path.join(output_dir, "result")

    subprocess.run(["mmseqs", "createdb", input_fasta, db], check=True)

    subprocess.run(
        [
            "mmseqs",
            "cluster",
            db,
            result,
            tmp_dir,
            "--min-seq-id",
            str(identity),
            "-c",
            "0.25",
            "-s",
            "10",
            "--alignment-mode",
            "3",
            "--seq-id-mode",
            "1",
            "--mask",
            "0",
            "--cov-mode",
            "2",
        ],
        check=True,
    )

    subprocess.run(
        [
            "mmseqs",
            "createtsv",
            db,
            db,
            result,
            os.path.join(output_dir, "cluster_map.tsv"),
        ],
        check=True,
    )

    return os.path.join(output_dir, "cluster_map.tsv")


def parse_cluster_tsv(tsv_path: str) -> Dict[str, List[str]]:
    cluster_dict = {}
    with open(tsv_path, "r") as f:
        for line in f:
            rep, member = line.strip().split("\t")
            if rep not in cluster_dict:
                cluster_dict[rep] = []
            cluster_dict[rep].append(member)
    return cluster_dict
