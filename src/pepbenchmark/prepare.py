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
from typing import List, Optional

import pandas as pd

from pepbenchmark.analyze.data_analyzer import (
    analyze_class_dataset,
    analyze_regression_dataset,
)
from pepbenchmark.pep_utils.neg_sample import NegSampler
from pepbenchmark.pep_utils.redundancy import Redundancy
from pepbenchmark.preprocess import preprocess_dataset
from pepbenchmark.raw_data import DATASET_MAP
from pepbenchmark.utils.deduplication import deduplicate_single
from pepbenchmark.utils.logging import get_logger
from pepbenchmark.utils.outlier_removal import remove_outliers_by_sequence

logger = get_logger()


class DataPrepare:
    def __init__(
        self,
        dataset_name: str,
        base_data_dir: Optional[str] = None,
        filter_length: Optional[int] = 50,
    ):
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

        self.type = DATASET_MAP.get(dataset_name, {}).get("type", None)
        self.filter_length = filter_length
        if self.type == "binary_classification":
            self.load_raw_pos_seqs()
        else:
            self.load_raw_data()

    def load_raw_pos_seqs(self) -> List[str]:
        pos_path = os.path.join(self.dataset_path, "pos_seqs.csv")
        if not os.path.exists(pos_path):
            raise FileNotFoundError(
                f"File not found: {pos_path} \n"
                f"Please ensure the dataset path is correct."
            )
        pos_df = pd.read_csv(pos_path)
        seqs = pos_df["sequence"].tolist()
        seqs = list(set(seqs))
        if self.filter_length:
            seqs = [s for s in seqs if len(s) <= self.filter_length]
        self.pos_seqs = seqs
        return self.pos_seqs

    def load_raw_data(self):
        data_path = os.path.join(self.dataset_path, "origin_data.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"File not found: {data_path}\n"
                f"Please ensure the dataset path is correct."
            )
        df = pd.read_csv(data_path)
        df["length"] = df["sequence"].apply(len)
        if self.filter_length:
            df = df[df["length"] <= self.filter_length]

        self.seqs = df["sequence"].tolist()
        self.labels = df["label"].tolist()

        return self.seqs, self.labels


class ClassDataPrepare(DataPrepare):
    def __init__(
        self,
        dataset_name: str,
        base_data_dir: Optional[str] = None,
        filter_length: Optional[int] = 50,
    ):
        type = DATASET_MAP.get(dataset_name, {}).get("type", None)
        if type != "binary_classification":
            raise ValueError(
                f"Dataset {dataset_name} is not a binary classification dataset."
            )
        super().__init__(dataset_name, base_data_dir, filter_length)

    def filt_redundancy(
        self,
        dedup_method: str = "mmseqs",
        identity: float = 0.9,
        processes: Optional[int] = None,
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
        self,
        user_sampling_pool_path: Optional[str] = None,
        filt_length: Optional[int] = None,
        dedup_identity: Optional[float] = None,
        ratio: float = 1,
        random_seed: int = 42,
        processes: Optional[int] = None,
    ):
        sampler = NegSampler(
            self.dataset_name,
            user_sampling_pool_path,
            filt_length,
            dedup_identity,
            processes=processes,
        )
        neg_seqs = sampler.get_sample_result(
            self.pos_seqs, ratio, limit="length", seed=random_seed
        )

        sampler.get_sample_info()
        sampler.visualization(plot_type="kde")
        self.neg_seqs = neg_seqs
        return neg_seqs

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

    def validate_data(
        self,
        processes: Optional[int] = None,
        enable_visualization: bool = True,
        save_plots: bool = True,
        plot_save_dir: Optional[str] = None,
    ):
        analyze_class_dataset(
            dataset_name=self.dataset_name,
            base_data_dir=self.dataset_path,
            processes=processes,
            enable_visualization=enable_visualization,
            save_plots=save_plots,
            plot_save_dir=plot_save_dir,
        )


class RegressionDataPrepare(DataPrepare):
    def __init__(
        self,
        dataset_name: str,
        base_data_dir: Optional[str] = None,
        filter_length: Optional[int] = 50,
    ):
        type = DATASET_MAP.get(dataset_name, {}).get("type", None)
        if type != "regression":
            raise ValueError(f"Dataset {dataset_name} is not a regression dataset.")
        super().__init__(dataset_name, base_data_dir, filter_length)

    def remove_outliers(
        self, method: str = "zscore", threshold: float = 1.5, z_threshold: float = 3.0
    ) -> tuple[List[str], List[float]]:
        """
        Remove outliers for regression tasks.

        Args:
            method (str): Outlier detection method, supports "iqr" or "zscore".
            threshold (float): Threshold multiplier for IQR method, default is 1.5.
            z_threshold (float): Threshold for Z-score method, default is 3.0.

        Returns:
            tuple: (List of sequences after removing outliers, List of labels after removing outliers)
        """
        # Create DataFrame for processing
        df = pd.DataFrame({"sequence": self.seqs, "label": self.labels})

        # Use utility function to remove outliers
        df_clean, outlier_indices = remove_outliers_by_sequence(
            df=df,
            sequence_column="sequence",
            label_column="label",
            method=method,
            threshold=threshold,
            z_threshold=z_threshold,
        )

        # Update instance variables
        self.seqs = df_clean["sequence"].tolist()
        self.labels = df_clean["label"].tolist()

        return self.seqs, self.labels

    def deduplicate(self) -> tuple[List[str], List[float]]:
        """
        Remove completely duplicate sequences.

        Returns:
            tuple: (List of deduplicated sequences, List of deduplicated labels)
        """
        # Create DataFrame
        df = pd.DataFrame({"sequence": self.seqs, "label": self.labels})

        # Use utility function for deduplication
        df_dedup, stats = deduplicate_single(df, self.type)

        # Update instance variables
        self.seqs = df_dedup["sequence"].tolist()
        self.labels = df_dedup["label"].tolist()

        return self.seqs, self.labels

    def validate_data(
        self,
        enable_visualization: bool = True,
        save_plots: bool = True,
        plot_save_dir: Optional[str] = None,
    ):
        analyze_regression_dataset(
            dataset_name=self.dataset_name,
            base_data_dir=self.dataset_path,
            print_summary=True,
            enable_visualization=enable_visualization,
            save_plots=save_plots,
            plot_save_dir=plot_save_dir,
        )

    def save_processed_data(self):
        df = pd.DataFrame({"sequence": self.seqs, "label": self.labels})
        df.to_csv(os.path.join(self.dataset_path, "processed_data.csv"), index=False)


def prepare_processed_class_data(
    dataset_name="BBP",
    dedup_method="mmseqs",
    dedup_identity=0.9,
    neg_sample_ratio=1,
    neg_sampling_pool_path=None,
    filter_length=None,
    validate_data=False,
    processes: Optional[int] = None,
    enable_visualization: bool = True,
    save_plots: bool = True,
    plot_save_dir: Optional[str] = "./new_plots",
    random_seed: int = 42,
):
    logger.info(f"Preparing dataset {dataset_name}")
    dataprepare_tool = ClassDataPrepare(dataset_name=dataset_name)
    dataprepare_tool.filt_redundancy(
        dedup_method=dedup_method,
        identity=dedup_identity,
        processes=processes,
        visualization=False,
    )
    dataprepare_tool.sample_neg_seqs(
        user_sampling_pool_path=neg_sampling_pool_path,
        ratio=neg_sample_ratio,
        filt_length=filter_length,
        dedup_identity=dedup_identity,
        random_seed=random_seed,
        processes=processes,
    )

    dataprepare_tool.save_combine_csv()
    if validate_data:
        dataprepare_tool.validate_data(
            processes=processes,
            enable_visualization=enable_visualization,
            save_plots=save_plots,
            plot_save_dir=plot_save_dir,
        )

    # Calculate the official features
    # preprocess_dataset(dataset_name=dataset_name, save_results=True)


def prepare_processed_regression_data(
    dataset_name="E.coli_mic",
    filter_length=None,
    outlier_remove_method="iqr",
    validate_data=True,
    enable_visualization: bool = True,
    save_plots: bool = True,
    plot_save_dir: Optional[str] = "./regression_data_plots",
):
    logger.info(f"Preparing dataset {dataset_name}")
    dataprepare_tool = RegressionDataPrepare(
        dataset_name=dataset_name, filter_length=filter_length
    )
    dataprepare_tool.load_raw_data()
    dataprepare_tool.remove_outliers(method=outlier_remove_method)
    dataprepare_tool.deduplicate()
    dataprepare_tool.save_processed_data()
    if validate_data:
        dataprepare_tool.validate_data(
            enable_visualization=enable_visualization,
            save_plots=save_plots,
            plot_save_dir=plot_save_dir,
        )
    # Calculate the official features
    preprocess_dataset(dataset_name=dataset_name, save_results=True)


if __name__ == "__main__":
    dataset_names = list(DATASET_MAP.keys())

    dataset_names = [
        "bbp",
        "nonfouling",
        "antibacterial",
        "antifungal",
        "antiviral",
        "antimicrobial",
        "E.coli_mic",
        "P.aeruginosa_mic",
        "S.aureus_mic",
        "antiparasitic",
        "ace_inhibitory",
        "anticancer",
        "antidiabetic",
        "antioxidant",
        "neuropeptide",
        "quorum_sensing",
        "ttca",
        "solubility",
        "hemolytic",
        "hemolytic_hc50",
        "toxicity",
        "allergen",
        "antiinflamatory",
        "antiaging",
        "anti_mammalian_cell",
        "dppiv_inhibitors",
        "cpp",
    ]

    for dataset_name in dataset_names:
        prepare_processed_class_data(dataset_name=dataset_name)
