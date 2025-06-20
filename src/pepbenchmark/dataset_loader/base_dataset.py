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

import logging
import warnings
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd
from metadata import DATASET_MAP
from utils.augmentation import (
    combine,
    random_delete,
    random_insertion_with_a,
    random_replace,
    random_replace_with_a,
    random_swap,
)
from utils.split import cold_split, homology_based_split, random_split
from visualization.distribution import (
    plot_peptide_distribution,
    plot_peptide_distribution_spited,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore")


class BaseDataset:
    """base data loader class that contains functions shared by almost all data loader classes."""

    def __init__(self):
        pass

    def get_data(self, format: str = "df"):
        """
        Arguments:
            format (str, optional): the dataset format [dict, df]
            split (str, optional): The type of split to use ('Random_split', 'Homology_based_split'). If None, returns the path to 'combine.csv'.
            fold_seed (int, optional): The seed for the random split. Ignored if `split` is None.

        Returns:
            pd.DataFrame/dict[str, pd.DataFrame]

        Raises:
            AttributeError: format not supported

        """
        if format == "df":
            df = self.dataset.copy()
            return df
        elif format == "dict":
            return {
                "peptide": self.sequence,
                "Y": self.y,
            }

    def get_split(
        self,
        method="random",
        seed=42,
        frac=None,
        seq_alighment="mmseq2",
        sim_threshold=None,
        visualization=False,
    ):
        frac = [0.8, 0.1, 0.1] if frac is None else frac
        df = self.get_data(format="df")

        if method == "random":
            split = random_split(df, frac=frac, seed=seed)
        elif method == "homology_based_split":
            split = homology_based_split(
                df,
                frac=frac,
                seq_alighment=seq_alighment,
                sim_threshold=sim_threshold,
                seed=seed,
            )
        elif method == "cold_split":
            split = cold_split(df, frac=frac, sim_threshold=sim_threshold, seed=seed)
        else:
            raise NotImplementedError(f"Split method {method} not implemented.")

        self.split = split
        self.split_ready = True
        return split

    def deduplicate(
        self, mode: str = "average", dedup_keys: Optional[List[str]] = None
    ) -> pd.DataFrame:
        dedup_keys = dedup_keys
        df = self.dataset.copy()

        if self.type == "regression":
            grouped = df.groupby(dedup_keys, as_index=False)
            if mode == "average":
                df = grouped["label"].mean().reset_index()
            elif mode == "min":
                df = grouped["label"].min().reset_index()
            elif mode == "max":
                df = grouped["label"].max().reset_index()
            else:
                raise NotImplementedError(f"Mode {mode} not implemented.")

        elif self.type == "binary_classification":

            def resolve_conflict(subdf):
                label_counts = Counter(subdf["label"])
                if len(label_counts) == 1:
                    return subdf.iloc[[0]]
                elif mode == "remove_all":
                    return pd.DataFrame(columns=subdf.columns)
                elif mode == "majority":
                    majority_label = label_counts.most_common(1)[0][0]
                    majority_rows = subdf[subdf["label"] == majority_label]
                    return majority_rows.iloc[[0]]
                else:
                    raise ValueError(f"Unsupported mode '{mode}' for classification")

            df = (
                df.groupby(dedup_keys, as_index=False)
                .apply(resolve_conflict)
                .reset_index(drop=True)
            )

        self.dataset = df
        self.sequence = self.dataset["sequence"].tolist()
        self.label = self.dataset["label"]
        return self

    def augmentation(self, method: str = "random_replace", ratio: float = 0.02):
        assert method in [
            "random_replace",
            "random_delete",
            "random_replace_with_a",
            "random_swap",
            "random_insertion_with_a",
        ], f"Invalid method: {method}"
        if method == "random_replace":
            new_inputs, new_labels = random_replace(self.sequence, self.label, ratio)
        elif method == "random_delete":
            new_inputs, new_labels = random_delete(self.sequence, self.label, ratio)
        elif method == "random_replace_with_a":
            new_inputs, new_labels = random_replace_with_a(
                self.sequence, self.label, ratio
            )
        elif method == "random_swap":
            new_inputs, new_labels = random_swap(self.sequence, self.label, ratio)
        elif method == "random_insertion_with_a":
            new_inputs, new_labels = random_insertion_with_a(
                self.sequence, self.label, ratio
            )

        self.sequence, self.label = combine(
            self.sequence, self.label, new_inputs, new_labels
        )
        self.dataset = pd.concat(
            [pd.DataFrame(self.sequence, columns=["sequence"]), self.label], axis=1
        )
        return self

    def binarize(self, threshold: float = 100, order: str = "descending"):
        if self.type == "binary_classification":
            raise NotImplementedError(
                "Binarization not implemented for binary classification."
            )
        if threshold is None:
            raise AttributeError(
                "Please specify the threshold to binarize the data by "
                "'binarize(threshold = N)'!"
            )

        if len(np.unique(self.label)) == 2:
            logger.info("The data is already binarized!")
        else:
            logger.info(
                "Binariztion using threshold "
                + str(threshold)
                + ", default, we assume the smaller values are 1 "
                "and larger ones is 0, you can change the order "
                "by 'binarize(order = 'ascending')'"
            )
            if order == "ascending":
                self.label = np.array(
                    [1 if i else 0 for i in np.array(self.label) > threshold]
                )
            elif order == "descending":
                self.label = np.array(
                    [1 if i else 0 for i in np.array(self.label) < threshold]
                )
            else:
                raise AttributeError(
                    "Please specify the order of binarization by "
                    "'binarize(order = 'ascending' or 'descending')'"
                )
            if (
                np.unique(self.label)
                .reshape(
                    -1,
                )
                .shape[0]
                < 2
            ):
                raise AttributeError("Adjust your threshold, there is only one class.")

        self.dataset = pd.concat(
            [pd.DataFrame(self.sequence, columns=["sequence"]), self.label], axis=1
        )

        return self

    def convert_to_log(self, form: str = "standard"):
        if form == "binding":
            self.label = -np.log10(self.label * 1e-9 + 1e-10)
        elif form == "standard":
            self.label = np.sign(self.label) * np.log(abs(self.label) + 1e-10)
        self.dataset = pd.concat(
            [pd.DataFrame(self.sequence, columns=["sequence"]), self.label], axis=1
        )
        return self

    def convert_from_log(self, form: str = "standard"):
        if form == "binding":
            self.label = (10 ** (-self.label) - 1e-10) / 1e-9
        elif form == "standard":
            sign = np.sign(self.label)
            self.label = sign * (np.exp(sign * self.label) - 1e-10)
        self.dataset = pd.concat(
            [pd.DataFrame(self.sequence, columns=["sequence"]), self.label], axis=1
        )
        return self

    def label_distribution(self):
        pass

    def pep_distribution(self):
        "Visualize the distribution of peptides in the dataset."

        if self.split_ready is False:
            plot_peptide_distribution(self.dataset, self.dataset_name, self.type)
        else:
            plot_peptide_distribution_spited(self.split, self.dataset_name, self.type)

    def print_stats(self):
        """print statistics"""
        logger.info(
            "There are "
            + str(len(self.sequence))
            + " sampels, and "
            + str(len(np.unique(self.sequence)))
            + " unique peptide sequences in the dataset."
        )

    def get_metadata(self):
        inform_dict = DATASET_MAP.get(self.dataset_name)
        logger.info(inform_dict)
