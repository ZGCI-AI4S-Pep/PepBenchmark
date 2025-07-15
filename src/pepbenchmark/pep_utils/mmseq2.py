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
from typing import Any, Dict, List, Optional

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from pepbenchmark.utils.logging import get_logger

logger = get_logger()


def save_fasta(fasta_list: List[str], path: str) -> None:
    records = [
        SeqRecord(Seq(seq), id=f"seq{i}", description="")
        for i, seq in enumerate(fasta_list)
    ]
    SeqIO.write(records, path, "fasta")


def _add_custom_params(cmd_cluster: List[str], params: Dict[str, Any]) -> None:
    custom_params = {
        k: v
        for k, v in params.items()
        if k
        not in [
            "coverage",
            "sensitivity",
            "alignment_mode",
            "seq_id_mode",
            "mask",
            "cov_mode",
            "threads",
            "max_iterations",
        ]
    }
    for key, value in custom_params.items():
        # Convert underscores to hyphens for MMseqs2 parameter format
        param_name = key.replace("_", "-")
        if isinstance(value, bool):
            if value:  # Only add flag if True
                cmd_cluster.append(f"--{param_name}")
        else:
            cmd_cluster.extend([f"--{param_name}", str(value)])


def _build_mmseqs_cluster_command(
    db: str,
    result: str,
    tmp_dir: str,
    identity: float,
    params: Dict[str, Any],
) -> List[str]:
    cmd_cluster = [
        "mmseqs",
        "cluster",
        db,
        result,
        tmp_dir,
        "--min-seq-id",
        str(identity),
    ]
    param_map = {
        "coverage": "-c",
        "sensitivity": "-s",
        "alignment_mode": "--alignment-mode",
        "seq_id_mode": "--seq-id-mode",
        "mask": "--mask",
        "cov_mode": "--cov-mode",
        "threads": "--threads",
        "max_iterations": "--max-iterations",
    }
    for key, flag in param_map.items():
        if key in params:
            cmd_cluster.extend([flag, str(params[key])])

    _add_custom_params(cmd_cluster, params)
    return cmd_cluster


def run_mmseqs_clustering(
    input_fasta: str,
    output_dir: str,
    tmp_dir: str,
    identity: float,
    **mmseqs_kwargs: Any,
) -> str:
    """
    Run MMseqs2 clustering on the input FASTA file.

    Args:
        input_fasta: Path to input FASTA file
        output_dir: Directory for output files
        tmp_dir: Temporary directory for MMseqs2
        identity: Minimum sequence identity threshold (--min-seq-id)
        **mmseqs_kwargs: Additional MMseqs2 parameters:
            - coverage: Coverage threshold (-c, default: 0.25)
            - sensitivity: Sensitivity (-s, default: 10)
            - alignment_mode: Alignment mode (--alignment-mode, default: 3)
            - seq_id_mode: Sequence identity mode (--seq-id-mode, default: 1)
            - mask: Mask (--mask, default: 0)
            - cov_mode: Coverage mode (--cov-mode, default: 2)
            - threads: Number of threads (--threads, default: None)
            - max_iterations: Maximum iterations (--max-iterations, default: None)
            - Additional parameters can be passed as key-value pairs

    Returns:
        str: Path to the generated TSV file
    """
    # Default MMseqs2 parameters
    default_params = {
        "coverage": 0.25,
        "sensitivity": 10,
        "alignment_mode": 3,
        "seq_id_mode": 1,
        "mask": 0,
        "cov_mode": 2,
    }

    # Update defaults with provided kwargs
    params = {**default_params, **mmseqs_kwargs}

    # Validate parameters
    validate_mmseqs_params(params)

    logger.info(f"MMseqs2 clustering parameters: identity={identity}, params={params}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    db = os.path.join(output_dir, "db")
    result = os.path.join(output_dir, "result")

    # Step 1: createdb
    cmd_createdb = ["mmseqs", "createdb", input_fasta, db]
    logger.info("Running command: " + " ".join(cmd_createdb))
    subprocess.run(
        cmd_createdb,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Step 2: cluster
    cmd_cluster = _build_mmseqs_cluster_command(db, result, tmp_dir, identity, params)

    logger.info("Running command: " + " ".join(cmd_cluster))
    subprocess.run(
        cmd_cluster,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )

    # Step 3: createtsv
    tsv_path = os.path.join(output_dir, "cluster_map.tsv")
    cmd_tsv = ["mmseqs", "createtsv", db, db, result, tsv_path]
    logger.info("Running command: " + " ".join(cmd_tsv))
    subprocess.run(
        cmd_tsv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )

    return tsv_path


def parse_cluster_tsv(tsv_path: str) -> Dict[str, List[str]]:
    cluster_dict = {}
    with open(tsv_path, "r") as f:
        for line in f:
            rep, member = line.strip().split("\t")
            if rep not in cluster_dict:
                cluster_dict[rep] = []
            cluster_dict[rep].append(member)
    return cluster_dict


def _validate_param(
    params: Dict[str, Any],
    key: str,
    valid_range: Any,
    message: str,
) -> Optional[str]:
    if key in params:
        value = params[key]
        if isinstance(valid_range, tuple):
            if not (valid_range[0] <= value <= valid_range[1]):
                return message.format(value)
        elif value not in valid_range:
            return message.format(value)
    return None


def validate_mmseqs_params(params):
    """
    Validate MMseqs2 parameters and provide warnings for potentially problematic values.

    Args:
        params: Dictionary of MMseqs2 parameters
    """
    param_validations = {
        "coverage": ((0.0, 1.0), "Coverage should be between 0.0 and 1.0, got {}"),
        "sensitivity": (
            (1.0, 20.0),
            "Sensitivity should typically be between 1.0 and 20.0, got {}",
        ),
        "alignment_mode": (
            [0, 1, 2, 3],
            "Alignment mode should be 0, 1, 2, or 3, got {}",
        ),
        "seq_id_mode": (
            [0, 1, 2],
            "Sequence identity mode should be 0, 1, or 2, got {}",
        ),
        "cov_mode": ([0, 1, 2], "Coverage mode should be 0, 1, or 2, got {}"),
    }

    warnings = []
    for key, (valid_range, message) in param_validations.items():
        warning = _validate_param(params, key, valid_range, message)
        if warning:
            warnings.append(warning)

    if "threads" in params:
        threads = params["threads"]
        if not isinstance(threads, int) or threads < 1:
            warnings.append(f"Threads should be a positive integer, got {threads}")

    if warnings:
        logger.warning("MMseqs2 parameter warnings: " + "; ".join(warnings))

    return len(warnings) == 0
