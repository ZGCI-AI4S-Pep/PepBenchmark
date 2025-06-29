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

import bisect
from abc import ABC
from itertools import chain
from pathlib import Path
from typing import Iterable, Union

from pep.entity import Residue
from pep.utils import aspath
from rdkit.Chem import PandasTools


class Library(ABC):
    """Base class for all libraries."""

    def __init__(self, id: str):
        self.id = id
        self.child_dict = dict()

    def __len__(self):
        return len(self.child_dict)

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, size={len(self)})"

    def __contains__(self, id):
        return id in self.child_dict

    def __getitem__(self, key: Union[str, int]) -> Residue:
        raise NotImplementedError()

    def __iter__(self):
        """Iterate over the library."""
        for key in self.child_dict:
            yield self.child_dict[key]


class MonomerLibrary(Library):
    """Monomer library for BILN inputs."""

    def __init__(self, id: str):
        super().__init__(id)

    @classmethod
    def from_sdf_file(cls, id: str, sdf_path: Union[str, Path]) -> "MonomerLibrary":
        """Create a MonomerLibrary from an SDF file."""
        instance = cls(id)
        df_group = PandasTools.LoadSDF(aspath(sdf_path))
        for idx, row in df_group.iterrows():
            monomer = Residue.from_monomer_info(row)
            instance.child_dict[monomer.m_abbr] = monomer
        return instance

    @classmethod
    def from_monomer_list(
        cls, id: str, monomer_list: Iterable[Residue]
    ) -> "MonomerLibrary":
        """Create a MonomerLibrary from a list of Residue objects."""
        instance = cls(id)
        for monomer in monomer_list:
            if not isinstance(monomer, Residue):
                raise TypeError(f"Expected Residue, got {type(monomer)}")
            if monomer.m_abbr in instance.child_dict:
                raise ValueError(f"Duplicate monomer abbreviation: {monomer.m_abbr}")
            instance.child_dict[monomer.m_abbr] = monomer
        return instance

    def __getitem__(self, key: Union[str, int]) -> Residue:
        """Get monomer by key (monomer name or index)."""
        if isinstance(key, str):
            if key not in self:
                raise KeyError(f"Monomer {key} not found in library.")
            return self.child_dict[key]
        elif isinstance(key, int):
            if key < 0:
                if -key > len(self):
                    raise ValueError(
                        "absolute value of index should not exceed dataset length"
                    )
                key = len(self) + key
            return self[list(self.child_dict.keys())[key]]
        elif isinstance(key, slice):
            keys = list(self.child_dict.keys())
            return [self[keys[i]] for i in range(*key.indices(len(keys)))]
        else:
            raise TypeError("Key must be a string or an integer.")


class ConcatLibrary(Library):
    def __init__(self, id: str, libs: Iterable[Library]) -> None:
        super().__init__(id)
        self.child_dict = {l.id: l for l in libs}
        assert len(self.child_dict) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        for d in self.child_dict.values():
            assert not isinstance(
                d, Library
            ), "ConcatDataset only accepts Library objects"
        self.cumulative_sizes = self.cumsum(self.libs)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: Union[int, str]) -> Residue:
        if isinstance(idx, str):
            for lib in self.child_dict.values():
                if idx in lib:
                    return lib[idx]
            else:
                raise KeyError(f"Monomer {idx} not found in library.")
        elif isinstance(idx, int):
            if idx < 0:
                if -idx > len(self):
                    raise ValueError(
                        "absolute value of index should not exceed dataset length"
                    )
                idx = len(self) + idx
            lib_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if lib_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[lib_idx - 1]
            return list(self.child_dict.values())[lib_idx][sample_idx]
        else:
            raise TypeError("Key must be a string or an integer.")

    def __contains__(self, id):
        for lib in self.child_dict.values():
            if id in lib.monomer_dict:
                return True
        return False

    def __iter__(self):
        yield from chain(self.child_dict.values())
