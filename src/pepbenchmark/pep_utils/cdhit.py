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

from pepbenchmark.utils.logging import get_logger

logger = get_logger()


def _add_custom_params_cdhit(cmd_cluster: List[str], params: Dict[str, Any]) -> None:
    custom_params = {
        k: v
        for k, v in params.items()
        if k not in ["local_alignment", "aln_coverage", "tolerant"]
    }
    for key, value in custom_params.items():
        # Convert underscores to hyphens for MMseqs2 parameter format
        param_name = key.replace("_", "-")
        if isinstance(value, bool):
            if value:  # Only add flag if True
                cmd_cluster.append(f"--{param_name}")
        else:
            cmd_cluster.extend([f"--{param_name}", str(value)])


def _build_cdhit_cluster_command(
    input_fasta_path: str, result_path: str, identity: float, params: Dict[str, Any]
) -> List[str]:
    cmd_cluster = [
        "cd-hit",
        "-i",
        input_fasta_path,
        "-o",
        result_path,
        "-c",
        str(identity),
        "-n",
        "2",
        "-d",
        "0",
        "-l",
        "1",
        "-g",
        "1",
    ]

    param_map = {"local_alignment": "-G", "aln_coverage": "-aL", "tolerant": "-t"}

    for key, flag in param_map.items():
        if key in params:
            cmd_cluster.extend([flag, str(params[key])])

    _add_custom_params_cdhit(cmd_cluster, params)

    return cmd_cluster


def run_cdhit_clustering(
    input_fasta: str, output_dir: str, identity: float, **cdhit_kwargs
) -> str:
    """
    Run CD-HIT clustering on the input FASTA file.

    Args:
        input_fasta: Path to input FASTA file
        output_dir: Directory for output files
        identity: Sequence identity threshold (-c)
        **cdhit_kwargs: Additional CD-HIT parameters
            - local_alignment: Local alignment (-G, default: 0)
            - aln_coverage: Alignment coverage for the longer sequence (-aL, default: 0.7)
            - tolerant: Tolerant (default: 0)

    Returns:
        str: Path to the clustered FASTA file
    """
    default_params = {
        "local_alignment": 0,
        "aln_coverage": 0.7,
        "tolerant": 0,  # 0% redundant sequences allowed
    }
    params = {**default_params, **cdhit_kwargs}

    validate_cdhit_params(params)
    logger.info(f"cdhit clustering parameters: identity={identity}, params={params}")

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "peptides_cdhit.fasta")

    cmd_cluster = _build_cdhit_cluster_command(
        input_fasta, result_path, identity, params
    )

    logger.info("Running CD-HIT command: " + " ".join(cmd_cluster))
    subprocess.run(
        cmd_cluster, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )

    return result_path


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


def validate_cdhit_params(params: Dict[str, Any]) -> bool:
    """
    Validate cdhit parameters and provide warnings for potentially problematic values.

    Args:
        params: Dictionary of cdhit parameters
    """
    param_validations = {
        "identity": (
            (0.4, 1.0),
            "Identity threshold should be between 0.4 and 1.0, got {}",
        ),
        "local_alignment": (
            [0, 1],
            "Local alignment (L) should be between 0 and 1, got {}",
        ),
        "aln_coverage": (
            (0.0, 1.0),
            "Alignment coverage (aL) should be between 0.0 and 1.0, got {}",
        ),
        "tolerant": ((0, 100), "Tolerant (t) should be between 0 and 100, got {}"),
    }

    warnings = []
    for key, (valid_range, message) in param_validations.items():
        warning = _validate_param(params, key, valid_range, message)
        if warning:
            warnings.append(warning)

    if "local_alignment" in params and "aln_coverage" in params:
        local_align = params["local_alignment"]
        if local_align != 0:
            warnings.append(
                "Local alignment (L) should be 0 when using alignment coverage (aL)"
            )

    if warnings:
        logger.warning("CD-HIT parameter warnings: " + "; ".join(warnings))

    return len(warnings) == 0


def parse_cdhit_clstr(path: str) -> Dict[str, List[str]]:
    """
    Parse CD-HIT .clstr file to get clustered sequences.
    """
    cluster_dict = {}
    clstr_path = path + ".clstr"
    with open(clstr_path, "r") as f:
        cluster_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                if cluster_lines:
                    _add_cluster_entry(cluster_lines, cluster_dict)
                cluster_lines = []
            else:
                cluster_lines.append(line)
        if cluster_lines:
            _add_cluster_entry(cluster_lines, cluster_dict)
    return cluster_dict


def _add_cluster_entry(
    cluster_lines: List[str], cluster_dict: Dict[str, List[str]]
) -> None:
    rep_seq = None
    members = []
    for seq in cluster_lines:
        seq_id = seq.split(">")[1].split("...")[0]
        if seq.endswith("*"):
            rep_seq = seq_id
        members.append(seq_id)
    if rep_seq is None:
        logger.error("No representative sequence found in cluster!")
    cluster_dict[rep_seq] = members


def get_representative_ids_cdhit(path: str) -> List[str]:
    """
    Get representative sequence IDs from CD-HIT .clstr file.
    """
    rep_ids = []
    file_path = path + ".clstr"
    with open(file_path, "r") as f:
        for line in f:
            if line.strip().endswith("*"):
                seq_id = line.split(">")[1].split("...")[0]
                rep_ids.append(seq_id)
    return rep_ids


def get_representative_seqs(path: str) -> List[str]:
    """
    Get representative sequences.
    """
    rep_seqs = []
    for record in SeqIO.parse(path, "fasta"):
        rep_seqs.append(str(record.seq))
    return rep_seqs
