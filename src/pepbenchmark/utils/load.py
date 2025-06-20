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

import pandas as pd
from metadata import get_dataset_path


def property_dataset_load(name, target):
    path = get_dataset_path(name)
    df = pd.read_csv(path)

    if target is None:
        target = df.columns.tolist()[1:]
    elif isinstance(target, str):
        target = [target]

    if not all(col in df.columns for col in target):
        raise ValueError(f"Target column(s) {target} not found in dataset {name}.")

    else:
        non_empty_mask = (df[target] != 0).any(axis=1)
        df = df[non_empty_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "The dataset is empty after filtering for non-null target values."
        )

    return df


# ---------- Loading pretrained models  ----------
"""
AVAILABLE_MODELS = {
    'esm2_t6_8M_UR50D': 320,
    'prot_bert': 1024,
}

def load_model(model: str):
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model} not available. Available models: {list(AVAILABLE_MODELS.keys())}")
    if model.lower() == 'esm2':
        lab = 'facebook'
        model = 'esm2_t6_8M_UR50D'
    elif model.lower() == 'prot_bert':
        lab = 'Rostlab'
        model = 'prot_bert'


    model = AutoModel.from_pretrained(f'{lab}/{model}',
                                      trust_remote_code=True)
                                      """
