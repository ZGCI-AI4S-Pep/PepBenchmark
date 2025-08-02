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


from pepbenchmark.metadata import DATASET_MAP
from pepbenchmark.raw_data import (
    get_current_official_data_version,
    set_official_data_version,
)

AVALIABLE_DATASET = list(DATASET_MAP.keys())


class DatasetManager:
    def __init__(
        self,
        dataset_name: str,
        dataset_dir: str = None,
        force_download: bool = False,
        data_version: str = None,
    ):
        self.dataset_name = dataset_name
        self._check_official_dataset_name()

        # Set official dataset if specified
        if data_version is not None:
            set_official_data_version(data_version)

        self.dataset_metadata = DATASET_MAP.get(self.dataset_name)
        if self.dataset_metadata is None:
            raise ValueError(f"Dataset metadata not found for: {self.dataset_name}")

        self.dataset_dir = (
            dataset_dir if dataset_dir else self.dataset_metadata.get("path")
        )
        self.force_download = force_download
        self.official_data_version = get_current_official_data_version()

    def _check_official_dataset_name(self) -> bool:
        """Checks if the dataset name exists in DATASET_MAP."""
        if self.dataset_name not in AVALIABLE_DATASET:
            raise ValueError(
                f"Unknown dataset name: {self.dataset_name}. "
                f"Available datasets: {AVALIABLE_DATASET}"
            )
        else:
            return True
