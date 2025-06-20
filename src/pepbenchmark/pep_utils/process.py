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

import numpy as np
import pandas as pd
from pep_utils.convert import Peptide
from pep_utils.featurizer import PeptideFeaturizer
from tqdm.notebook import tqdm

tqdm.pandas(desc="Converting sequences")


def pepconvert(df, original_format, convert_format, feature_type, all_nature):
    #
    if convert_format is not None:
        df[convert_format] = df["sequence"].apply(
            lambda s: Peptide(s, format=original_format, all_nature=all_nature).to(
                convert_format
            )
        )

    if feature_type is not None:
        featurizer = PeptideFeaturizer(
            input_format=original_format,
            feature_type=feature_type,
            is_natural=all_nature,
        )
        features = featurizer(df["sequence"].tolist())

        if isinstance(features, list):
            df = pd.concat([df.reset_index(drop=True), pd.DataFrame(features)], axis=1)
        elif isinstance(features, np.ndarray):
            for i in range(features.shape[1]):
                df[f"feat_{i}"] = features[:, i]
        elif isinstance(features, pd.DataFrame):
            df = pd.concat([df.reset_index(drop=True), features], axis=1)
    return df
