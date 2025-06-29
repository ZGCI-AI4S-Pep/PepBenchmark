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

from collections.abc import Iterable as IterableABC
from typing import Any, Dict, Iterable, List, Optional, Union

from pepbenchmark.pep_utils.convert import AVALIABLE_TRANSFORM
from pepbenchmark.utils.logging import get_logger

logger = get_logger()


class PeptideFeaturizer:
    def __init__(
        self,
        input_format: str,
        output_format: str,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.param = params or {}
        self.input_format = input_format
        self.output_format = output_format
        key = (self.input_format.lower(), self.output_format.lower())
        if key not in AVALIABLE_TRANSFORM:
            raise ValueError(
                f"Transform from {self.input_format} to {self.output_format} is not supported"
            )

        transform_cls = AVALIABLE_TRANSFORM[key]

        try:
            self.transformer = transform_cls(**self.param)
            logger.info(
                f"Successfully instantiated {transform_cls.__name__} with parameters: {self.param}"
            )
        except TypeError as e:
            logger.info(f"Failed to instantiate {transform_cls.__name__}. Reason: {e}")
            logger.info("Expected parameters (from docstring):")
            logger.info(transform_cls.__doc__)

    def __call__(self, sequences: Union[str, Iterable[str]]) -> Union[Any, List[Any]]:
        """
        Support both single sequence (str) and batch of sequences (Iterable[str]).
        """
        if isinstance(sequences, str):
            return self.transformer(sequences)
        elif isinstance(sequences, IterableABC) and not isinstance(sequences, str):
            return [self.transformer(seq) for seq in sequences]
        else:
            raise TypeError(
                f"Input must be a string or a list of strings, got {type(sequences)}"
            )


if __name__ == "__main__":
    featurizer = PeptideFeaturizer(
        "smiles", "fingerprint", {"fp_type": "Morgan", "nBits": 1024}
    )
    result = featurizer("CC(=O)OC1=CC=CC=C1C(=O)O")
