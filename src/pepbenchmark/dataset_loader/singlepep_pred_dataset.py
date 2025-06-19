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

import warnings

import pandas as pd

from src.pepbenchmark.metadata import get_dataset_path

from ..metadata import DATASET_MAP
from ..utils import cold_split, homology_based_split, random_split
from .base_dataset import BaseDataset

warnings.filterwarnings("ignore")


class SingleTaskDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str,
        convert_format=None,
        feature_type=None,
    ):
        inform_dict = DATASET_MAP.get(dataset_name)
        path = get_dataset_path(dataset_name)
        self.dataset_name = dataset_name
        self.convert_format = convert_format
        self.feature_type = feature_type
        self.raw_format = inform_dict["format"].lower()
        self.all_nature = inform_dict["nature"] == "natural"
        self.type = inform_dict["type"]
        self.dataset = pd.read_csv(path)
        self.sequence = self.dataset["sequence"].tolist()
        self.label = self.dataset["label"]
        self.split_ready = False

    def get_data(self, format="df", split=None, fold_seed=1, split_ready=True):
        """
        Arguments:
            format (str, optional): the dataset format
            split (str, optional): The type of split to use ('Random_split', 'Homology_based_split'). If None, returns the path to 'combine.csv'.
            fold_seed (int, optional): The seed for the random split. Ignored if `split` is None.
            split_ready (bool, optional): Whether to retrive split readythe dataset or not. If True, returns a dictionary of train, val, and test dataframes. If False, returns the path to 'combine.csv'.


        Returns:
            pd.DataFrame/dict[str, pd.DataFrame]

        Raises:
            AttributeError: format not supported
        """

        if (self.convert_format is not None) or (self.convert_result is None):
            from ..pep_utils import PepConvert

        self.split_ready = split_ready

        if split_ready == False:
            df = self.dataset.copy()
            df = PepConvert(
                df,
                self.raw_format,
                self.convert_format,
                self.feature_type,
                self.all_nature,
            )
            return df

        else:
            assert split in [
                "Random_split",
                "Homology_based_split",
            ], "split should be 'Random_split' or 'Homology_based_split'."

            assert fold_seed in [1, 2, 3, 4, 5], "fold_seed should be 1, 2, 3, 4, or 5."

            print("Loading split ready dataset")
            dict_split = {}
            for split_type in ["train", "valid", "test"]:
                df_split = pd.read_csv(
                    get_dataset_path(
                        self.dataset_name, split=split, fold_seed=1, type=split_type
                    )
                )
                dict_split[split_type] = PepConvert(
                    df_split,
                    self.raw_format,
                    self.convert_format,
                    self.feature_type,
                    self.all_nature,
                )

            self.split = dict_split
            return dict_split

    def get_split(
        self,
        method="random",
        seed=42,
        frac=[0.8, 0.1, 0.1],
        seq_alighment="mmseq2",
        sim_threshold=None,
        visualization=False,
    ):
        df = self.get_data(format="df", split_ready=False)

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


class MutiTaskDataset(BaseDataset):
    """def __init__(self,
                 dataset_name: str,
                 label_name = None,
                 convert_format = None,
                 feature_type = None,
                 ):

                         if (dataset_name in natural_multiclass_keys) or (dataset_name in non_natural_multiclass_keys):
            if label_name is None:
                raise ValueError(
                    "Please select a label name.. You can use pepbenchmark.utils.retrieve_label_name_list('"
                    + dataset_name +"') to retrieve all available label names.")

    self.dataset = property_dataset_load(dataset_name, label_name)
        self.sequence = self.dataset['sequence'].tolist()
        self.label = self.dataset['label']
        self.split_ready = False"""
