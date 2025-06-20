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
import sys
import warnings

from dataset_loader.singlepep_pred_dataset import SingleTaskDataset

warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


data = SingleTaskDataset(
    "QS_APML",
    convert_format="smiles",
    feature_type="ecfp",
)

data.get_metadata()
data.pep_distribution()
train = data.get_split(method="random")["train"]
data.print_stats()
data.pep_distribution()
