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

import re
from dataclasses import dataclass, field
from typing import Dict, Tuple

from pep.base import Parser, Serializer
from pep.entity import ConnectionInfo, ParsedData, ParsedMonomerInfo
from pep.library import MonomerLibrary


# biln格式的解析和序列化
class BilnSyntaxError(ValueError):
    """Error during BILN string parsing (syntax).在BILN字符串解析过程中抛出语法错误，便于定位和处理异常"""

    pass


@dataclass
class ParsedBilnMonomerInfo(ParsedMonomerInfo):
    """Holds parsed information about a single monomer instance."""

    explicit_connections: list[tuple[int, int]] = field(
        default_factory=list
    )  # List of (bond_id, r_group_idx)
    # 用于记录BILN解析过程中每个单体的详细连接和分隔符信息。
    preceded_by_hyphen: bool = False
    followed_by_hyphen: bool = False


class BilnParser(Parser):
    """Parses a BILN string into an intermediate representation.
    负责将BILN字符串解析为中间数据结构"""

    def __init__(self, monomer_library: MonomerLibrary):
        self._simple_monomer_pattern = r"([A-Za-z0-9_]+)"  # Allow underscore
        self._bracketed_monomer_pattern = r"\[([^\]]+?)\]"
        self._monomer_pattern = re.compile(
            f"({self._bracketed_monomer_pattern}|{self._simple_monomer_pattern})"
        )
        self._connection_pattern = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
        super().__init__(monomer_library)

    def parse(self, biln_string: str) -> ParsedData:
        """Parses the BILN string."""
        if not biln_string:
            raise BilnSyntaxError("Empty BILN string provided for parsing.")

        parsed_data, curr_info = ParsedData(), None
        pos, m_cnt = 0, 0
        last_component_was_monomer, expecting_monomer = False, True

        while pos < len(biln_string):
            # --- 1. Try Monomer ---
            # Match monomer abbreviation (either simple or bracketed)
            monomer_match = self._monomer_pattern.match(biln_string, pos)
            if monomer_match and expecting_monomer:
                abbr, full_match_str = self._extract_monomer_abbr(monomer_match, pos)
                try:
                    residue = self.library[abbr]
                except KeyError:
                    raise BilnSyntaxError(
                        f"Monomer abbreviation '{abbr}' (at pos {pos}) not found in the library."
                    )
                curr_info = ParsedBilnMonomerInfo(m_idx=m_cnt, abbr=abbr, res=residue)
                m_cnt += 1

                # Check preceding separator
                if (
                    not last_component_was_monomer
                    and pos > 0
                    and biln_string[pos - 1] == "-"
                ):
                    curr_info.preceded_by_hyphen = True
                    if parsed_data.monomers:
                        # Mark the previous monomer as followed by hyphen
                        parsed_data.monomers[-1].followed_by_hyphen = True

                parsed_data.monomers.append(curr_info)
                pos += len(full_match_str)
                last_component_was_monomer = True
                expecting_monomer = False
                continue  # Check for connections or separators

            # --- 2. Try Connection ---
            # Can only appear after a monomer
            if last_component_was_monomer and biln_string[pos] == "(":
                conn_match = self._connection_pattern.match(biln_string, pos)
                if not conn_match:
                    raise BilnSyntaxError(
                        f"Invalid connection format starting at position {pos}"
                    )
                bond_id, r_group_idx = self._parse_connection_details(
                    conn_match, curr_info, pos
                )
                # Record on monomer and in global map
                curr_info.explicit_connections.append((bond_id, r_group_idx))
                connection = ConnectionInfo(m_idx=curr_info.m_idx, rg_idx=r_group_idx)
                parsed_data.explicit_bond_map[bond_id].append(connection)
                pos += conn_match.end() - conn_match.start()
                # Still last_component_was_monomer = True, expecting separator or EOF
                continue

            # --- 3. Try Separator ---
            if last_component_was_monomer:
                separator = biln_string[pos]
                if separator in [".", "-"]:
                    pos += 1
                    last_component_was_monomer = False
                    expecting_monomer = True
                    # If '-', the following monomer will handle marking flags
                    continue
                # If it wasn't a separator or connection, fall through to error

            # --- 4. Error ---
            self._raise_unexpected_sequence(biln_string, pos, expecting_monomer)

        # --- Add Implicit Connections ---
        for i in range(len(parsed_data.monomers) - 1):
            m1, m2 = parsed_data.monomers[i], parsed_data.monomers[i + 1]
            if m1.followed_by_hyphen and m2.preceded_by_hyphen:
                # Standard peptide bond: R2 (COOH) from m1 --- R1 (NH2) from m2
                implicit_bond1 = ConnectionInfo(m_idx=m1.m_idx, rg_idx=1)
                implicit_bond2 = ConnectionInfo(m_idx=m2.m_idx, rg_idx=0)
                parsed_data.implicit_bond_map[i].append(implicit_bond1)
                parsed_data.implicit_bond_map[i].append(implicit_bond2)

        # --- Final Syntax Checks ---
        self._validate_parsing_end_state(
            biln_string, last_component_was_monomer, parsed_data
        )
        self._validate_bond_id_pairing(parsed_data.explicit_bond_map)

        return parsed_data

    def _extract_monomer_abbr(self, match: re.Match, pos: int) -> Tuple[str, str]:
        """Extracts monomer abbreviation from regex match."""
        full_match_str = match.group(0)
        bracketed_content = match.group(2)
        simple_content = match.group(3)

        if bracketed_content is not None:
            abbr = bracketed_content.strip()
            # if not abbr:
            #     raise BilnSyntaxError(f"Empty brackets found at position {pos}")
            if "[" in abbr or "]" in abbr:
                raise BilnSyntaxError(
                    f"Nested or invalid brackets within monomer '{full_match_str}' at pos {pos}"
                )
        elif simple_content is not None:
            abbr = simple_content
        else:
            # Just placeholder, should not happen
            raise BilnSyntaxError(
                f"Monomer pattern matched but no content found at pos {pos}"
            )
        return abbr, full_match_str

    def _parse_connection_details(
        self, conn_match: re.Match, monomer_info: ParsedBilnMonomerInfo, pos: int
    ) -> Tuple[int, int]:
        """Parses and validates bond_id and r_group from connection match."""
        bond_id_str, r_group_str = conn_match.groups()
        bond_id = int(bond_id_str)
        r_group_input = int(r_group_str)  # 1-based from BILN
        r_group_idx = r_group_input - 1  # Convert to 0-based index for internal use
        # Validate R-group index against the specific monomer's definition
        if not (
            0 <= r_group_idx < len(monomer_info.res.m_Rgroups)
            and monomer_info.res.m_Rgroups[r_group_idx] is not None
        ):
            raise BilnSyntaxError(
                f"Monomer '{monomer_info.abbr}' (instance {monomer_info.m_idx}) does not have a valid R-group definition "
                f"for R{r_group_input} (index {r_group_idx}). Available R-groups: {sum([i[0] is not None for i in monomer_info.res.m_Rgroups])}"
            )

        return bond_id, r_group_idx

    def _raise_unexpected_sequence(
        self, biln_string: str, pos: int, expecting_monomer: bool
    ):
        """Raises error for invalid character sequence."""
        found = f"'{biln_string[pos:]}'" if pos < len(biln_string) else "end of string"
        if expecting_monomer:
            if pos < len(biln_string) and biln_string[pos] == "(":
                raise BilnSyntaxError(
                    f"Connection found at position {pos} without a preceding monomer."
                )
            else:
                raise BilnSyntaxError(
                    f"Expected a monomer abbreviation but found {found} at position {pos}"
                )
        else:
            raise BilnSyntaxError(
                f"Expected a separator ('.', '-'), connection '(...)', or end of string, but found {found} at position {pos}"
            )

    def _validate_parsing_end_state(
        self,
        biln_string: str,
        last_component_was_monomer: bool,
        parsed_data: ParsedData,
    ):
        """Checks if the BILN string ended correctly."""
        if not last_component_was_monomer:
            # Allow single monomer string like "A" which won't have flags set
            is_multi_component = (
                any(
                    m.followed_by_hyphen
                    or m.preceded_by_hyphen
                    or m.explicit_connections
                    for m in parsed_data.monomers
                )
                or "." in biln_string
            )  # Crude check for separators
            if is_multi_component or len(parsed_data.monomers) > 1:
                raise BilnSyntaxError(
                    "BILN string cannot end with a separator ('.' or '-')"
                )
            # Single monomer case is okay

    def _validate_bond_id_pairing(self, explicit_bond_map: Dict[int, list]):
        """Checks if all explicit bond IDs appear exactly twice."""
        invalid_bond_ids = []
        for bond_id, connections in explicit_bond_map.items():
            if len(connections) != 2:
                invalid_bond_ids.append((bond_id, len(connections)))
        if invalid_bond_ids:
            error_msg = ", ".join(
                [f"ID {bid} ({count} times)" for bid, count in invalid_bond_ids]
            )
            raise BilnSyntaxError(
                f"Incorrect explicit bond ID pairings: {error_msg}. Each ID must appear exactly twice."
            )


class BilnSerializer(Serializer):
    def __init__(self, library: MonomerLibrary):
        super().__init__(library)
        self._bond_serial = 1

    def serialize(self, parsed_data: ParsedData) -> str:
        """Convert parsed data back to a BILN string."""
        # reset bond serial number
        self._bond_serial = 1
        biln_parts, biln_str = [], []
        # [monomer_idx, [explicit_bonds...], seperator]
        for i, m in enumerate(parsed_data.monomers):
            assert i == m.m_idx, f"Monomer index mismatch: {i} != {m.m_idx}"
            biln_parts.append([m.m_idx, [], ""])

        # implicit connections
        for bidx, conns in parsed_data.implicit_bond_map.items():
            # conns[0] or [1]: m_idx, rg_idx
            assert (
                len(conns) == 2
            ), f"Invalid number of connections for bond ID {bidx}: {len(conns)}"
            b1, b2 = conns[0], conns[1]
            assert (
                (b2.m_idx - b1.m_idx) == 1
            ), f"Implicit connections {b1.m_idx} and {b2.m_idx} for bond ID {bidx} should be adjacent."
            assert (
                b1.rg_idx == 1 and b2.rg_idx == 0
            ), f"Invalid R-group indices for bond ID {bidx}: {b1.rg_idx}, {b2.rg_idx}"
            biln_parts[b1.m_idx][2] = "-"

        # explicit connections
        for bidx, conns in parsed_data.explicit_bond_map.items():
            assert (
                len(conns) == 2
            ), f"Invalid number of connections for bond ID {bidx}: {len(conns)}"
            b1, b2 = conns[0], conns[1]
            biln_parts[b1.m_idx][1].append((self._bond_serial, b1.rg_idx + 1))
            biln_parts[b2.m_idx][1].append((self._bond_serial, b2.rg_idx + 1))
            self._bond_serial += 1

        # Now build the BILN string
        for idx, (m_idx, exp_conns, sep) in enumerate(biln_parts):
            m = parsed_data.monomers[m_idx]
            abbr = m.res.m_abbr
            if "-" in abbr:
                abbr = f"[{abbr}]"
            conns_str = "".join(
                [f"({bond_id},{rg_idx})" for bond_id, rg_idx in exp_conns]
            )
            if idx != len(biln_parts) - 1 and not sep:
                # Not the last monomer and not hyphen, add chain separator
                sep = "."
            biln_str.append(f"{abbr}{conns_str}{sep}")

        return "".join(biln_str)


if __name__ == "__main__":
    from pep.builder import MolBuilder
    from pep.library import MonomerLibrary
    from rdkit import Chem

    lib = MonomerLibrary.from_sdf_file(
        "test_library", "PepDB-main/pep/resources/monomers.sdf"
    )

    # biln = "A-G-K(1,3)-D-D.ac(1,2)"
    # biln = "D-T-H-F-P-I-C(1,3)-I-F-C(2,3)-C(3,3)-G-C(2,3)-C(4,3)-H-R-S-K-C(3,3)-G-M-C(4,3)-C(1,3)-K-T"
    biln = "I-V-S-A-V-K-K-I-V-D-F-L-G-G-L-A-S-P"
    # 用给定的单体库 lib 创建一个 BILN 解析器，然后用它解析 BILN 字符串 biln，最终得到结构化的解析结果 parsed_data
    parsed_data = BilnParser(lib).parse(biln)
    print(parsed_data)

    mol = MolBuilder(parsed_data).build()
    smiles = Chem.MolToSmiles(mol)
    print(smiles)

    """print(biln)
    serializer = BilnSerializer(lib)
    serializer.serialize(parsed_data)"""
