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
from typing import Any, Dict, List

import pandas as pd
import requests
from pepbenchmark.single_peptide.base_dataset import DatasetManager
from pepbenchmark.utils.logging import get_logger

logger = get_logger()

EXCLUSIVE_MAP: Dict[str, List[str]] = {
    "TTCA_TCAHybrid": ["Immunomodulatory", "Immunoregulator"],
    "QS_APML": [],
    "Antibacterial": ["Antibacterial"],
    "Antiviral": ["Antiviral"],
}  # TODO: To be complete

INCLUSIVE_MAP: Dict[str, List[str]] = {
    "Nonfouling": ["Hemolytic", "Antimicrobial"],
}  # TODO: To be complete


class NegSampler(object):
    """
    Base class for negative sampling strategies used in peptide dataset construction.

    This class defines the common interface and basic utilities for extracting
    positive sequences from a dataset, and serves as a parent class for implementing
    various negative sampling strategies.
    """

    def __init__(self, dataset: DatasetManager) -> None:
        self.dataset = dataset
        self.dataset_name = dataset.dataset_name
        if (
            self.dataset_name not in EXCLUSIVE_MAP
            and self.dataset_name not in INCLUSIVE_MAP
        ):
            raise ValueError(
                f"Negative sampling is not supported for the {self.dataset_name} dataset."
            )

    def get_pos_seqs(self) -> List[str]:
        if (
            "fasta" not in self.dataset.official_feature_dict
            or "label" not in self.dataset.official_feature_dict
        ):
            raise ValueError(
                "Dataset must set official 'fasta' and 'label' features first."
            )
        fasta_list = self.dataset.get_official_feature("fasta")
        label_list = self.dataset.get_official_feature("label")
        pos_seqs = [seq for seq, label in zip(fasta_list, label_list) if label == 1]
        return pos_seqs


class NegSampleFromBioactive(NegSampler):
    """
    Negative sampling from PeptidePedia
    """

    def __init__(self, dataset: DatasetManager) -> None:
        super().__init__(dataset=dataset)

    def __call__(self, ratio: float, seed: int) -> List[str]:
        pos_seqs = self.get_pos_seqs()
        bio_df = self._load_negative_pool()
        negative_pool_df = self.get_negative_pool(bio_df)
        self._get_mapping(self.dataset_name)

        pos_df = pd.DataFrame({"sequence": pos_seqs})
        pos_df["length"] = pos_df["sequence"].apply(len)

        # create length bins
        bins = range(0, pos_df["length"].max() + 1, 1)
        pos_df["length_bin"] = pd.cut(pos_df["length"], bins=bins)
        bin_counts = pos_df["length_bin"].value_counts().sort_index()
        negative_pool_df["length"] = negative_pool_df["sequence"].apply(len)
        neg_seqs = self._negative_sample(
            negative_pool_df, bin_counts, ratio, random_seed=seed
        )
        logger.info(
            f"Get {len(neg_seqs)} negative samples from PeptidePedia,"
            f"Negative sampling with ratio {ratio} successfully!"
        )

        return neg_seqs

    def _download_negative_pool(self) -> None:
        url = "https://raw.githubusercontent.com/ZGCI-AI4S-Pep/peptide_data/main/multitask_peptidepedia.csv"
        negative_pool_path = os.path.join("negative_pool", "multitask_peptidepedia.csv")
        os.makedirs("negative_pool", exist_ok=True)
        logger.info(f"Downloading ===neg pool=== from {url}")
        try:
            with requests.get(url, stream=True, timeout=100) as r:
                r.raise_for_status()
                with open(negative_pool_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except requests.RequestException as e:
            logger.error(f"Failed to download negative_pool_path: {e}")
            raise

    def _load_negative_pool(self) -> pd.DataFrame:
        neg_path = os.path.join("negative_pool", "multitask_peptidepedia.csv")
        if not os.path.exists(neg_path):
            self._download_negative_pool()
        bio_df = pd.read_csv(neg_path)
        return bio_df

    def _get_mapping(self, dataset_name: str) -> None:
        if dataset_name in EXCLUSIVE_MAP:
            self.select = "exclusive"
            self.mapping = EXCLUSIVE_MAP[dataset_name]
        elif dataset_name in INCLUSIVE_MAP:
            self.select = "inclusive"
            self.mapping = INCLUSIVE_MAP[dataset_name]

    def get_negative_pool(self, bio_df: pd.DataFrame) -> pd.DataFrame:
        return self._get_negative_pool(bio_df)

    def _get_negative_pool(self, bio_df: pd.DataFrame) -> pd.DataFrame:
        bio_df.columns = [col.lower() for col in bio_df.columns]
        if self.select == "exclusive":
            excluded_activities = [a.lower() for a in self.mapping]
            mask = bio_df[excluded_activities].sum(axis=1) == 0
            negative_pool_df = bio_df[mask][["sequence"]].copy()
        elif self.select == "inclusive":
            included_activities = [a.lower() for a in self.mapping]
            mask = bio_df[included_activities].sum(axis=1) >= 1
            negative_pool_df = bio_df[mask][["sequence"]].copy()
        return negative_pool_df

    def _negative_sample(
        self,
        negative_pool_df: pd.DataFrame,
        bin_counts: Any,
        ratio: float,
        random_seed: int,
    ) -> List[str]:
        negative_pool_df = negative_pool_df.copy()
        negative_pool_df["used"] = False

        neg_samples = []

        for bin_range, pos_count in bin_counts.items():
            low = int(bin_range.left)
            high = int(bin_range.right)

            candidates: pd.DataFrame = negative_pool_df[
                (~negative_pool_df["used"])
                & (negative_pool_df["length"] > low)
                & (negative_pool_df["length"] <= high)
            ]

            neg_count = int(ratio * pos_count)

            sampled = candidates.sample(
                n=min(neg_count, len(candidates)), random_state=random_seed
            )
            negative_pool_df.loc[sampled.index, "used"] = True

            shortage = neg_count - len(sampled)

            step = 1
            while shortage > 0:
                next_low = high + step - 1
                next_high = high + step

                extra = negative_pool_df[
                    (~negative_pool_df["used"])
                    & (negative_pool_df["length"] > next_low)
                    & (negative_pool_df["length"] <= next_high)
                ]

                borrow_count = min(shortage, len(extra))
                if borrow_count == 0:
                    step += 1
                    continue

                borrowed = extra.sample(n=borrow_count, random_state=42 + step)
                negative_pool_df.loc[borrowed.index, "used"] = True
                sampled = pd.concat([sampled, borrowed])
                shortage -= borrow_count
                step += 1

            neg_samples.append(sampled)

        neg_df = pd.concat(neg_samples).drop_duplicates(subset="sequence")
        return neg_df["sequence"].tolist()


# 多任务数据集MultiTaskDatasetManager的实现形式待确定，可能后续与SingleTaskDatasetManager合并
"""
class MultiTaskNegSampler():
    def __init__(self, activity_list: List[str]):
        unknown_tasks = [task for task in activity_list if task not in EXCLUSIVE_MAP]
        if unknown_tasks:
            raise ValueError(f"Unknown task(s) in EXCLUSIVE_MAP: {unknown_tasks}")

        merged_exclude = set()
        for task in activity_list:
            merged_exclude.update(EXCLUSIVE_MAP[task])
        self.select = "exclusive"
        self.mapping = list(merged_exclude)"""
