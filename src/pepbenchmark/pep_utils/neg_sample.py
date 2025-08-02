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
import requests

from pepbenchmark.analyze.properties import (
    calculate_properties,
    compare_properties_distribution,
)
from pepbenchmark.analyze.visualization import visualize_property_distribution_compare
from pepbenchmark.raw_data import DATA_DIR
from pepbenchmark.utils.logging import get_logger

logger = get_logger()

EXCLUSIVE_MAP = {
    "ttca": ["immunomodulatory", "immunoregulator"],
    "quorum_sensing": ["quorum sensing"],
    "antibacterial": ["antibacterial"],
    "antiviral": ["antiviral"],
    "anticancer": ["anticancer", "cytotoxic", "anti mammalian cell"],
    "antifungal": ["anti fungal"],
    "antiparasitic": ["antiparasitic"],
    "antioxidant": ["anti oxidative"],
    "antimicrobial": ["antimicrobial"],
    "hemolytic": ["hemolytic", "cytotoxic", "anti mammalian cell"],
    "toxicity": ["toxic"],
    "neuropeptide": ["neuropeptide", "blood brain barrier penetrating", "neurological"],
    "antidiabetic": ["glucose"],
    "ace_inhibitory": ["coagulation + vascular", "angiotensinase inhibitor"],
    "bbp": [
        "blood brain barrier penetrating",
        "neuropeptide",
        "neurological",
        "cell penetrating",
    ],
    "allergen": ["allergen"],
    "antiinflamatory": ["immunological"],
    "antiaging": ["anti aging"],
    "dppiv_inhibitors": ["glucose"],
}  # TODO: To be complete

INCLUSIVE_MAP = {
    "Nonfouling": ["Hemolytic", "cytotoxic", "anti mammalian cell"],
}  # TODO: To be complete


BASE_URL = "https://raw.githubusercontent.com/ZGCI-AI4S-Pep/peptide_data/main/"


class NegSampler(object):
    """
    class for negative sampling strategies used in peptide dataset construction.

    Args:
        dataset_name: Name of the dataset.
        official_sampling_pool: Name of the official negative sampling pool.
    """

    def __init__(
        self,
        dataset_name: str,
        user_sampling_pool_path: Optional[str] = None,
        filt_length: Optional[int] = None,
        dedup_identity: Optional[float] = None,
        processes: Optional[int] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.filt_length = filt_length
        self.dedup_identity = dedup_identity
        self.processes = processes
        self.official_sampling_pool = None
        if user_sampling_pool_path:
            self.set_user_sampling_pool(user_sampling_pool_path)
        else:
            self.set_official_sampling_pool()
            self.user_sampling_pool = None
        self.sampling_pool = None
        self.pos_sequences = None
        self.neg_sequences = None
        if user_sampling_pool_path and (
            dataset_name in EXCLUSIVE_MAP or dataset_name in INCLUSIVE_MAP
        ):
            self.merge_pool = True
        else:
            self.merge_pool = False

    def get_sample_result(
        self,
        fasta: List[str],
        ratio: float,
        limit: Optional[str] = None,
        seed: int = 42,
    ) -> List[str]:
        if self.merge_pool:
            # When merging, make sure both official and user pools are loaded
            if self.official_sampling_pool is None:
                self.set_official_sampling_pool()
            pools = [
                pool
                for pool in [self.official_sampling_pool, self.user_sampling_pool]
                if pool is not None
            ]
            sampling_pool = (
                pd.concat(pools)
                .drop_duplicates(subset="sequence")
                .reset_index(drop=True)
            )
        elif self.user_sampling_pool is not None:
            # Use only the user pool, do not load the official pool
            sampling_pool = self.user_sampling_pool
        else:
            if self.official_sampling_pool is None:
                logger.info(
                    "No official sampling pool is set. Set official sampling pool from peptidepedia."
                )
                logger.info(
                    "If you want to use your own sampling pool, please set user_sampling_pool_path when initializing NegSampler."
                )
                self.set_official_sampling_pool()
            sampling_pool = self.official_sampling_pool

        if limit:
            pos_df = self._calculate_property_bins(fasta, limit, n_bins=10)
            neg_df = self._sample_by_property_bin(
                pos_df, sampling_pool, limit, ratio, random_seed=seed
            )
            pos_sequences = pos_df["sequence"].tolist()
            neg_sequences = neg_df["sequence"].tolist()
        else:
            logger.info("No limit is set. Sampling negative samples by random.")
            neg_sequences = self._sample_by_random(
                fasta, sampling_pool, ratio, random_seed=seed
            )
            pos_sequences = fasta

        self.pos_sequences = fasta
        self.neg_sequences = neg_sequences

        logger.info(
            f"Get {len(neg_sequences)} negative samples from sampling pool. ratio between negative and positive samples is {len(neg_sequences)/len(pos_sequences)} now. "
        )

        return neg_sequences

    def get_sample_info(self, properties: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate distribution difference scores between positive and negative samples.

        Args:
            properties: List of property names to evaluate. If None, auto-detect.

        Returns:
            DataFrame summarizing distribution difference metrics.
        """
        if not self.pos_sequences or not self.neg_sequences:
            raise ValueError(
                "No sample result is available. Run get_sample_result() first."
            )

        result = compare_properties_distribution(
            self.pos_sequences, self.neg_sequences, properties
        )
        logger.info(
            f"Distribution difference between positive and negative samples:\n{result}"
        )
        return result

    def visualization(
        self,
        properties: Optional[List[str]] = None,
        plot_type: str = "kde",
        bins: int = 20,
    ):
        """
        Visualize property distributions between positive and negative samples as a single figure with subplots.

        Args:
            properties: List of property names to visualize. If None, auto-detect.
            plot_type: 'hist' for histogram, 'kde' for density plot.
            bins: Number of bins for histogram.
        """
        if not self.pos_sequences or not self.neg_sequences:
            logger.error(
                "No sample result is available. Run get_sample_result() first."
            )
            return None
        visualize_property_distribution_compare(
            self.pos_sequences,
            self.neg_sequences,
            properties=properties,
            plot_type=plot_type,
            bins=bins,
        )

    def set_user_sampling_pool(self, user_sampling_pool_path: str) -> None:
        pool = self._get_user_sampling_pool(user_sampling_pool_path)
        if self.filt_length is not None or self.dedup_identity is not None:
            pool = self._filter_sequences(
                pool,
                length_limit=self.filt_length,
                identity=self.dedup_identity,
                processes=self.processes,
            )
        self.user_sampling_pool = pool
        logger.info(f"Set user sampling_pool: {user_sampling_pool_path} successfully")

    def set_official_sampling_pool(self) -> None:
        pool = self.get_official_negative_pool("peptidepedia")
        if self.filt_length is not None or self.dedup_identity is not None:
            pool = self._filter_sequences(
                pool,
                length_limit=self.filt_length,
                identity=self.dedup_identity,
                processes=self.processes,
            )
        self.official_sampling_pool = pool
        logger.info("Set official sampling_pool: peptidepedia successfully")

    def _download_negative_pool(self, sampling_pool: str) -> None:
        url = BASE_URL + f"{sampling_pool}.csv"
        negative_pool_path = os.path.join(
            f"{DATA_DIR}/negative_pool", f"{sampling_pool}.csv"
        )
        print(negative_pool_path)
        os.makedirs(f"{DATA_DIR}/negative_pool", exist_ok=True)
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

    def _load_negative_pool(self, sampling_pool: str) -> pd.DataFrame:
        neg_path = os.path.join(f"{DATA_DIR}/negative_pool", f"{sampling_pool}.csv")
        if not os.path.exists(neg_path):
            self._download_negative_pool(sampling_pool)
        neg_df = pd.read_csv(neg_path)
        return neg_df

    def _get_mapping(self, dataset_name: str) -> None:
        if dataset_name in EXCLUSIVE_MAP:
            self.select = "exclusive"
            self.mapping = EXCLUSIVE_MAP[dataset_name]
        elif dataset_name in INCLUSIVE_MAP:
            self.select = "inclusive"
            self.mapping = INCLUSIVE_MAP[dataset_name]
        else:
            self.select = None
            self.mapping = []

    def _get_user_sampling_pool(self, user_sampling_pool_path: str) -> pd.DataFrame:
        df = pd.read_csv(user_sampling_pool_path)
        if "sequence" not in df.columns:
            raise ValueError("User sampling pool CSV must contain a 'sequence' column.")
        pool = calculate_properties(df["sequence"].tolist())

        return pool

    def _filter_sequences(
        self,
        df: pd.DataFrame,
        length_limit: Optional[int] = None,
        identity: Optional[float] = None,
        processes: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Filter sequences longer than length_limit, and optionally perform deduplication.
        Args:
            df: DataFrame containing a 'sequence' column.
            length_limit: Maximum allowed sequence length. Sequences longer than this will be filtered out. None means no length filtering.
            identity: Identity threshold for deduplication. If None, no deduplication is performed.
            processes: Number of processes to use for deduplication (if applicable).
        Returns:
            Filtered DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame({"sequence": df})
        if length_limit is not None:
            mask = df["sequence"].astype(str).apply(lambda x: len(x) <= length_limit)
            df = df[mask].copy()
        if identity is not None:
            from pepbenchmark.pep_utils.redundancy import Redundancy

            rd = Redundancy()
            seqs = df["sequence"].tolist()
            deduped_seqs = rd.deduplicate(
                seqs,
                dedup_method="mmseqs",
                identity=identity,
                processes=processes if processes is not None else 16,
                visualization=False,
            )
            mask = df["sequence"].astype(str).isin(deduped_seqs)
            df = df[mask].copy()
        return df

    def get_official_negative_pool(self, sampling_pool: str) -> pd.DataFrame:
        neg_df = self._load_negative_pool(sampling_pool)
        neg_df.columns = [col.lower() for col in neg_df.columns]
        if sampling_pool == "peptidepedia":
            self._get_mapping(self.dataset_name)
            if self.select == "exclusive":
                excluded_activities = [a.lower() for a in self.mapping]
                mask = neg_df[excluded_activities].sum(axis=1) == 0
                neg_pool_df = neg_df[mask][["sequence"]].copy()
            elif self.select == "inclusive":
                included_activities = [a.lower() for a in self.mapping]
                mask = neg_df[included_activities].sum(axis=1) >= 1
                neg_pool_df = neg_df[mask][["sequence"]].copy()
            else:
                neg_pool_df = neg_df[["sequence"]].copy()
        elif sampling_pool == "uniprot_inbioactive":
            neg_pool_df = neg_df

        # 只保留 sequence 列，确保类型
        neg_pool_df = pd.DataFrame({"sequence": neg_pool_df["sequence"].tolist()})
        sampling_pool_df = calculate_properties(neg_pool_df["sequence"].tolist())
        return sampling_pool_df

    def _calculate_property_bins(
        self, sequences: List[str], property: str, n_bins: int
    ) -> pd.DataFrame:
        """
        Automatically determine bin edges and assign bin labels.

        Args:
            sequences: List of peptide sequences.
            property: Property name to bin.
            n_bins: Number of bins per property.
        """
        assert (
            property in ["mw", "hydrophobicity", "charge", "isoelectricpoint", "length"]
        ), "Unsupported property! property must be one of ['mw', 'hydrophobicity', 'charge', 'isoelectricpoint', 'length']"
        df = calculate_properties(sequences)

        prop_values = df[property].dropna()
        unique_count = prop_values.nunique()

        if unique_count <= n_bins:
            bins = sorted(prop_values.unique().tolist() + [prop_values.max() + 1])
            df[f"{property}_bin"] = pd.cut(df[property], bins=bins, include_lowest=True)
        else:
            df[f"{property}_bin"] = pd.qcut(df[property], q=n_bins, duplicates="drop")

        return df

    def _sample_by_property_bin(
        self,
        pos_df: pd.DataFrame,
        sampling_pool: pd.DataFrame,
        limit: str,
        ratio: float,
        random_seed: int,
    ) -> pd.DataFrame:
        bin_counts = pos_df[f"{limit}_bin"].value_counts().sort_index()

        sampling_pool_df = sampling_pool.copy()
        sampling_pool_df["used"] = False

        neg_samples = []
        bin_ranges = bin_counts.index.tolist()
        pos_seq = pos_df["sequence"].tolist()

        for idx, bin_range in enumerate(bin_ranges):
            # bin_range may be an interval or an int (if unique_count <= n_bins)
            if hasattr(bin_range, "left") and hasattr(bin_range, "right"):
                low = bin_range.left
                high = bin_range.right
            else:
                low = bin_range
                high = bin_range
            pos_count = bin_counts[bin_range]

            candidates = sampling_pool_df[
                (~sampling_pool_df["used"])
                & (sampling_pool_df[f"{limit}"] > low)
                & (sampling_pool_df[f"{limit}"] <= high)
            ]
            if "sequence" in candidates.columns:
                candidates = candidates[~candidates["sequence"].isin(pos_seq)]

            neg_count = int(ratio * pos_count)

            if len(candidates) > 0:
                sampled = candidates.sample(
                    n=min(neg_count, len(candidates)), random_state=random_seed
                )
            else:
                sampled = pd.DataFrame(columns=sampling_pool_df.columns)
            sampling_pool_df.loc[sampled.index, "used"] = True

            shortage = neg_count - len(sampled)

            # Borrow only from the next bin if available
            next_idx = idx + 1
            while shortage > 0 and next_idx < len(bin_ranges):
                next_bin_range = bin_ranges[next_idx]
                if hasattr(next_bin_range, "left") and hasattr(next_bin_range, "right"):
                    next_low = next_bin_range.left
                    next_high = next_bin_range.right
                else:
                    next_low = next_bin_range
                    next_high = next_bin_range

                extra = sampling_pool_df[
                    (~sampling_pool_df["used"])
                    & (sampling_pool_df[f"{limit}"] > next_low)
                    & (sampling_pool_df[f"{limit}"] <= next_high)
                ]
                if "sequence" in extra.columns:
                    extra = extra[~extra["sequence"].isin(pos_seq)]
                borrow_count = min(shortage, len(extra))
                if borrow_count > 0:
                    borrowed = extra.sample(
                        n=borrow_count, random_state=random_seed + idx + 1
                    )
                    sampling_pool_df.loc[borrowed.index, "used"] = True
                    sampled = pd.concat([sampled, borrowed])
                    shortage -= borrow_count
                next_idx += 1

            neg_samples.append(sampled)

        if neg_samples:
            neg_df = pd.concat(neg_samples).drop_duplicates(subset="sequence")
        else:
            neg_df = pd.DataFrame(columns=pd.Index(["sequence"]))
        return neg_df

    def _sample_by_random(
        self,
        pos_sequences: List[str],
        sampling_pool: pd.DataFrame,
        ratio: float,
        random_seed: int,
    ) -> List[str]:
        neg_count = int(ratio * len(pos_sequences))
        sampling_pool_df = sampling_pool.copy()
        if len(sampling_pool_df) == 0:
            return []
        neg_df = sampling_pool_df.sample(
            n=min(neg_count, len(sampling_pool_df)), random_state=random_seed
        )
        return neg_df["sequence"].tolist()


if __name__ == "__main__":
    from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager

    dataset_name = "bbp"  # Change this to your dataset name

    dataset_manager = SingleTaskDatasetManager(
        dataset_name=dataset_name, official_feature_names=["fasta"]
    )

    positive_sequences = dataset_manager.get_positive_sequences()
    sampler = NegSampler(dataset_name=dataset_name)

    #  sampling by property bin
    neg_seqs = sampler.get_sample_result(
        positive_sequences, ratio=1.0, limit="length", seed=123
    )

    print("Sampled negative sequences:")
    for seq in neg_seqs[:10]:
        print(seq)

    sampler.get_sample_info()
    sampler.visualization(plot_type="kde")
