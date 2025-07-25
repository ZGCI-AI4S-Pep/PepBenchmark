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
from typing import List

import pandas as pd

from pepbenchmark.pep_utils.neg_sample import NegSampler
from pepbenchmark.pep_utils.redundancy import Redundancy
from pepbenchmark.preprocess import preprocess_dataset
from pepbenchmark.raw_data import DATASET_MAP
from pepbenchmark.utils.logging import get_logger

logger = get_logger()


class DataPrepare:
    def __init__(self, dataset_name: str, base_data_dir: str = None):
        self.dataset_name = dataset_name

        # Validate dataset name
        if dataset_name not in DATASET_MAP:
            available_datasets = list(DATASET_MAP.keys())
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASET_MAP. "
                f"Available datasets: {available_datasets}"
            )

        # Use base_data_dir if provided, otherwise use the path from metadata
        if not base_data_dir:
            self.dataset_path = DATASET_MAP.get(dataset_name, {}).get("path", None)
        else:
            self.dataset_path = base_data_dir

    def load_raw_pos_seqs(self, filter_length: bool = False) -> List[str]:
        pos_path = os.path.join(self.dataset_path, "pos_seqs_50.csv")
        if not os.path.exists(pos_path):
            raise FileNotFoundError(
                f"File not found: {pos_path} \n"
                f"Please ensure the dataset path is correct."
            )
        pos_df = pd.read_csv(pos_path)
        seqs = pos_df["sequence"].tolist()
        if filter_length:
            seqs = [s for s in seqs if len(s) <= 50]
        self.pos_seqs = seqs
        return self.pos_seqs

    def filt_redundancy(
        self,
        dedup_method: str = "mmseqs",
        identity: float = 0.9,
        processes=None,
        visualization=False,
    ):
        rd = Redundancy()
        remain_seqs = rd.deduplicate(
            self.pos_seqs,
            dedup_method=dedup_method,
            identity=identity,
            processes=processes,
            visualization=visualization,
        )
        self.pos_seqs = remain_seqs
        return self.pos_seqs

    def sample_neg_seqs(
        self, official_sampling_pool: str, ratio: float, random_seed: int = 42
    ):
        sampler = NegSampler(self.dataset_name, official_sampling_pool)
        neg_seqs = sampler.get_sample_result(
            self.pos_seqs, ratio, limit="length", seed=random_seed
        )
        sampler.get_sample_info()
        sampler.visualization(plot_type="kde")
        self.neg_seqs = neg_seqs
        return neg_seqs

    def set_neg_seqs(self, neg_seqs: List[str]):
        pass

    def load_raw_neg_seqs(self) -> List[str]:
        neg_path = os.path.join(self.dataset_path, "neg_seqs.csv")
        if not os.path.exists(neg_path):
            raise FileNotFoundError(f"File not found: {neg_path}")

        self.neg_seqs = pd.read_csv(neg_path, header=None)["sequence"].tolist()
        return self.neg_seqs

    def save_combine_csv(self):
        if self.neg_seqs is None:
            logger.warning("Negative sequences not found")
            return

        combine_df = (
            pd.DataFrame(
                {
                    "sequence": self.pos_seqs + self.neg_seqs,
                    "label": [1] * len(self.pos_seqs) + [0] * len(self.neg_seqs),
                }
            )
            .sample(frac=1)
            .reset_index(drop=True)
        )

        output_dir = self.dataset_path
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "combine.csv")
        combine_df.to_csv(output_path, index=False)
        logger.info(f"Saving combined sequences to {output_path}")

    def _balance_dataset(self):
        pass

    def validate_data(self):
        pass


def prepare_processed_data(
    dataset_name="BBP",
    dedup_method="mmseqs",
    identity=0.9,
    neg_sample_ratio=1,
    sampling_pool="peptidepedia",
    filter_length=True,
):
    dataprepare_tool = DataPrepare(dataset_name=dataset_name)

    dataprepare_tool.load_raw_pos_seqs(filter_length=filter_length)
    dataprepare_tool.filt_redundancy(
        dedup_method=dedup_method, identity=identity, processes=16, visualization=False
    )
    dataprepare_tool.sample_neg_seqs(
        official_sampling_pool=sampling_pool, ratio=neg_sample_ratio
    )
    dataprepare_tool.save_combine_csv()

    preprocess_dataset(dataset_name=dataset_name, save_results=True)


if __name__ == "__main__":
    dataset_names = list(DATASET_MAP.keys())
    dataset_names = ["antifungal", "antiviral", "quorum_sensing", "neuropeptide"]
    for dataset_name in dataset_names:
        dataset_name = dataset_name.lower()
        prepare_processed_data(dataset_name=dataset_name)
