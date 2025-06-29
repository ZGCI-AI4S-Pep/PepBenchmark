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

import numpy
import pandas as pd
from pepbenchmark.pep_utils.mmseq2 import (
    parse_cluster_tsv,
    run_mmseqs_clustering,
    save_fasta,
)
from pepbenchmark.utils.logging import get_logger

from .base_spliter import BaseSplitter

logger = get_logger()


class MMseqs2Spliter(BaseSplitter):
    def run(self, data, identity):
        with tempfile.TemporaryDirectory() as tmp_root:
            input_fasta = os.path.join(tmp_root, "input.fasta")
            output_dir = os.path.join(tmp_root, "output")
            tmp_dir = os.path.join(tmp_root, "tmp")

            # Save the input data to a FASTA file

            save_fasta(data, input_fasta)
            tsv_path = run_mmseqs_clustering(input_fasta, output_dir, tmp_dir, identity)
            cluster_map = parse_cluster_tsv(tsv_path)
            self._print_cluster_stats(cluster_map)

            return cluster_map

    def get_split_indices(
        self,
        data,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        identity=0.25,
        **kwargs,
    ):
        seed = kwargs.get("seed")
        numpy.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        cluster_map = self.run(data, identity)
        cluster_items = list(cluster_map.items())

        if seed is not None:
            numpy.random.RandomState(seed).shuffle(cluster_items)
        else:
            numpy.random.shuffle(cluster_items)

        train_data_size = int(len(data) * frac_train)
        valid_data_size = int(len(data) * frac_valid)
        test_data_size = int(len(data) * frac_test)

        train_ids, valid_ids, test_ids = [], [], []
        count_train = count_valid = count_test = 0

        for _, members in cluster_items:
            if count_train + len(members) <= train_data_size:
                train_ids.extend(members)
                count_train += len(members)
            elif count_valid + len(members) <= valid_data_size:
                valid_ids.extend(members)
                count_valid += len(members)
            else:
                test_ids.extend(members)
                count_test += len(members)

        logger.info(f"""Finish clustering and splitting data.
                    \nTarget train data size: {train_data_size}, Train: {len(train_ids)}
                    \nTarget valid data size：{valid_data_size}， Valid: {len(valid_ids)}
                    \nTarget test data size：{test_data_size}, Test: {len(test_ids)}""")

        # Create a mapping from sequence ID to index
        id_to_idx = {f"seq{i}": i for i in range(len(data))}
        return {
            "train": [id_to_idx[x] for x in train_ids if x in id_to_idx],
            "valid": [id_to_idx[x] for x in valid_ids if x in id_to_idx],
            "test": [id_to_idx[x] for x in test_ids if x in id_to_idx],
        }

    def get_split_kfold_indices(
        self,
        data,
        n_splits=5,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        identity=0.25,
        **kwargs,
    ):
        seed = kwargs.get("seed")
        spit_results = {}
        for i in range(n_splits):
            split_indices = self.get_split_indices(
                data,
                frac_train,
                frac_valid,
                frac_test,
                seed=seed + i if seed is not None else None,
            )
            spit_results[f"seed_{i}"] = split_indices
        return spit_results

    def _print_cluster_stats(self, cluster_map):
        log_message = f"Total clusters: {len(cluster_map)}\n"
        cluster_sizes = [len(members) for members in cluster_map.values()]

        bins = {
            "size = 1": 0,
            "size 2–4": 0,
            "size 5–9": 0,
            "size 10–19": 0,
            "size 20+": 0,
        }

        for size in cluster_sizes:
            if size == 1:
                bins["size = 1"] += 1
            elif 2 <= size <= 4:
                bins["size 2–4"] += 1
            elif 5 <= size <= 9:
                bins["size 5–9"] += 1
            elif 10 <= size <= 19:
                bins["size 10–19"] += 1
            else:
                bins["size 20+"] += 1

        log_message += "Cluster size distribution:\n"
        for label, count in bins.items():
            log_message += f"  {label}: {count} clusters\n"
        logger.info(log_message)


if __name__ == "__main__":
    import json

    from pepbenchmark.metadata import DATASET_MAP

    dataset_name = "Aox_APML"  # Change this to your dataset name
    path = DATASET_MAP.get(dataset_name).get("path")
    os.makedirs(f"{dataset_name}", exist_ok=True)
    df = pd.read_csv(path + "/combine.csv")

    split = MMseqs2Spliter()
    split_result = split.get_split_kfold_indices(
        df,
        n_splits=5,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        identity=0.25,
        seed=42,
    )
    with open(f"{dataset_name}/mmseqs2_split.json", "w") as f:
        json.dump(split_result, f, indent=4)
