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

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import pandas as pd
from rdkit import Chem


# --- Intermediate Data Structures ---
@dataclass
class ConnectionInfo:
    """Stores details about one side of a connection."""

    m_idx: int
    rg_idx: int  # 0-based index into Residue.m_Rgroups


@dataclass
class ParsedMonomerInfo:
    """Holds parsed information about a single monomer instance."""

    m_idx: int
    abbr: str
    res: "Residue"  # Reference to the library residue


@dataclass
class ParsedData:
    """Holds the complete result of parsing the BILN string."""

    monomers: List[ParsedMonomerInfo] = field(default_factory=list)
    # Maps bond_id to list of ConnectionInfo objects
    explicit_bond_map: Dict[int, List[ConnectionInfo]] = field(
        default_factory=lambda: defaultdict(list)
    )
    implicit_bond_map: Dict[int, List[ConnectionInfo]] = field(
        default_factory=lambda: defaultdict(list)
    )


class ResidueType(Enum):
    # TODO: use this to unify the residue types
    NAT = "natural"
    NNT = "non-natural"
    CAP = "cap"
    LINKER = "linker"

    def __str__(self):
        return self.value


class Residue:
    """Represents a residue. A Residue object stores atoms."""

    def __init__(self):
        """Initialize the class."""
        self.m_name: str = ""
        self.m_abbr: str = ""
        self.m_type: ResidueType = None
        self.m_Rgroups: List[str] = []
        self.natAnalog: str = ""
        self.pdbName: str = ""
        self.m_romol: Chem.Mol

    def __repr__(self):
        """Return the residue full id."""
        return f"<Res. {self.m_abbr} type={self.m_type} #Rgroups={sum([i[0] is not None for i in self.m_Rgroups])}>"

    @classmethod
    def from_monomer_info(cls, lib_df_row: pd.Series):
        res = cls()
        res.m_name = lib_df_row["m_name"]
        res.m_abbr = lib_df_row["m_abbr"]
        res.m_type = ResidueType(lib_df_row["m_type"])
        # TODO: filter out None values
        # Contents for Rgroups like OH, H,
        l1 = [
            v if v not in {"None", ""} else None
            for v in lib_df_row["m_Rgroups"].split(",")
        ]
        # Rgroup atom index in ROMol
        l2 = [
            int(v) if v not in {"None", ""} else None
            for v in lib_df_row["m_RgroupIdx"].split(",")
        ]
        # Attachment point index in ROMol
        l3 = [
            int(v) if v not in {"None", ""} else None
            for v in lib_df_row["m_attachmentPointIdx"].split(",")
        ]
        res.m_Rgroups = list(zip(l1, l2, l3))
        res.natAnalog = lib_df_row["natAnalog"]
        res.pdbName = lib_df_row["pdbName"]
        res.m_romol = lib_df_row["ROMol"]
        return res
