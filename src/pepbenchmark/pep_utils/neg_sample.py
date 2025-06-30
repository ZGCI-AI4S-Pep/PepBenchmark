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

from typing import List

import pandas as pd
from pepbenchmark.utils.logging import get_logger

logger = get_logger()

EXCLUSIVE_MAP = {
    "TTCA_TCAHybrid": ["Immunomodulatory", "Immunoregulator"],
    "QS_APML": [],
    "Antibacterial": ["Antibacterial"],
    "Antiviral": ["Antiviral"],
}  # TODO: To be complete

INCLUDE_MAP = {
    "Nonfouling": ["Hemolytic", "Antimicrobial"],
}  # TODO: To be complete

DATA_PATH = r"E:\pycharm\peptide_data\multitask_peptidepedia\combine.csv"


class NegSampler(object):
    def __init__(self, dataset_name: str):
        if dataset_name in EXCLUSIVE_MAP:
            self.select = "exclusive"
            self.mapping = EXCLUSIVE_MAP[dataset_name]
        elif dataset_name in INCLUDE_MAP:
            self.select = "inclusive"
            self.mapping = INCLUDE_MAP[dataset_name]
        else:
            raise ValueError(
                "Negative sampling is not supported for the given dataset name."
            )

    def __call__(self, pos_seq, ratio, seed):
        self.seed = seed
        return self._sample_from_bioactive(pos_seq, ratio, self.select, self.mapping)

    def _sample_from_bioactive(self, pos_seqs, ratio, select, mapping):
        bio_df = pd.read_csv(DATA_PATH)
        pos_df = pd.DataFrame({"sequence": pos_seqs})
        pos_df["length"] = pos_df["sequence"].apply(len)

        # create length bins
        bins = range(0, pos_df["length"].max() + 1, 1)
        pos_df["length_bin"] = pd.cut(pos_df["length"], bins=bins)
        bin_counts = pos_df["length_bin"].value_counts().sort_index()

        bio_df.columns = [col.lower() for col in bio_df.columns]

        if select == "exclusive":
            excluded_activities = [a.lower() for a in mapping]
            mask = bio_df[excluded_activities].sum(axis=1) == 0
            neg_pool_df = bio_df[mask][["sequence"]].copy()
        elif select == "inclusive":
            included_activities = [a.lower() for a in mapping]
            mask = bio_df[included_activities].sum(axis=1) >= 1
            neg_pool_df = bio_df[mask][["sequence"]].copy()

        neg_pool_df["length"] = neg_pool_df["sequence"].apply(len)
        neg_seqs = self._neg_sample(
            neg_pool_df, bin_counts, ratio, random_seed=self.seed
        )
        logger.info(f"Get {len(neg_seqs)} negative samples from PeptidePedia")
        return neg_seqs

    def _neg_sample(self, neg_pool_df, bin_counts, ratio, random_seed):
        neg_pool_df = neg_pool_df.copy()
        neg_pool_df["used"] = False  # 标记已使用的序列，防止重复采样

        neg_samples = []

        for bin_range, pos_count in bin_counts.items():
            low = int(bin_range.left)
            high = int(bin_range.right)

            candidates = neg_pool_df[
                (~neg_pool_df["used"])
                & (neg_pool_df["length"] > low)
                & (neg_pool_df["length"] <= high)
            ]

            neg_count = int(ratio * pos_count)

            sampled = candidates.sample(
                n=min(neg_count, len(candidates)), random_state=random_seed
            )
            neg_pool_df.loc[sampled.index, "used"] = True

            shortage = neg_count - len(sampled)

            step = 1
            while shortage > 0:
                next_low = high + step - 1
                next_high = high + step

                extra = neg_pool_df[
                    (~neg_pool_df["used"])
                    & (neg_pool_df["length"] > next_low)
                    & (neg_pool_df["length"] <= next_high)
                ]

                borrow_count = min(shortage, len(extra))
                if borrow_count == 0:
                    step += 1
                    continue

                borrowed = extra.sample(n=borrow_count, random_state=42 + step)
                neg_pool_df.loc[borrowed.index, "used"] = True
                sampled = pd.concat([sampled, borrowed])
                shortage -= borrow_count
                step += 1

            neg_samples.append(sampled)

        neg_df = pd.concat(neg_samples).drop_duplicates(subset="sequence")
        return neg_df["sequence"].tolist()


class MultiTaskNegSampler(NegSampler):
    def __init__(self, activity_list: List[str]):
        unknown_tasks = [task for task in activity_list if task not in EXCLUSIVE_MAP]
        if unknown_tasks:
            raise ValueError(f"Unknown task(s) in EXCLUSIVE_MAP: {unknown_tasks}")

        merged_exclude = set()
        for task in activity_list:
            merged_exclude.update(EXCLUSIVE_MAP[task])
        self.select = "exclusive"
        self.mapping = list(merged_exclude)
