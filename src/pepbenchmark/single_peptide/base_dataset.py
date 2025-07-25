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


from pepbenchmark.raw_data import DATASET_MAP

AVALIABLE_DATASET = list(DATASET_MAP.keys())


class DatasetManager:
    def __init__(
        self,
        dataset_name: str,
        dataset_dir: str = None,
        force_download: bool = False,
    ):
        self.dataset_name = dataset_name
        self._check_official_dataset_name()
        self.dataset_metadata = DATASET_MAP.get(self.dataset_name)
        self.dataset_dir = (
            dataset_dir if dataset_dir else self.dataset_metadata.get("path")
        )
        self.force_download = force_download

    def _check_official_dataset_name(self) -> bool:
        """Checks if the dataset name exists in DATASET_MAP."""
        if self.dataset_name not in AVALIABLE_DATASET:
            raise ValueError(
                f"Unknown dataset name: {self.dataset_name}. "
                f"Available datasets: {AVALIABLE_DATASET}"
            )
        else:
            return True
