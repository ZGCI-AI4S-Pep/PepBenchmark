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

import numpy
from pepbenchmark.utils.logging import get_logger

from .base_spliter import BaseSplitter

logger = get_logger(__name__)


class RandomSplitter(BaseSplitter):
    def get_split_indices(
        self, data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, **kwargs
    ):
        seed = kwargs.get("seed")
        numpy.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        if seed is not None:
            perm = numpy.random.RandomState(seed).permutation(len(data))
        else:
            perm = numpy.random.permutation(len(data))
        train_data_size = int(len(data) * frac_train)
        valid_data_size = int(len(data) * frac_valid)
        return {
            "train": perm[:train_data_size],
            "valid": perm[train_data_size : train_data_size + valid_data_size],
            "test": perm[train_data_size + valid_data_size :],
        }

    def get_split_kfold_indices(
        self, data, n_splits=5, frac_train=0.8, frac_valid=0.1, frac_test=0.1, **kwargs
    ):
        seed = kwargs.get("seed")
        split_results = {}
        for i in range(n_splits):
            split_indices = self.get_split_indices(
                data,
                frac_train,
                frac_valid,
                frac_test,
                seed=seed + i if seed is not None else None,
            )
            split_results[f"seed_{i}"] = split_indices
        logger.info(
            f"Generated {n_splits} random splits with seeds from {seed} to {seed + n_splits - 1 if seed is not None else 'None'}"
        )
        return split_results
