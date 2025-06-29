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


from pep.base import Parser, Serializer
from pep.builder import MolBuilder
from pep.entity import ConnectionInfo, ParsedData, ParsedMonomerInfo
from pep.library import Library, MonomerLibrary


class FastaParser(Parser):
    """
    Parser for FASTA files.
    """

    def __init__(self, monomer_library: Library):
        super().__init__(monomer_library)

    def parse(self, fasta: str) -> ParsedData:
        """
        Parse the input a FASTA sequence and return a ParsedData object.
        """
        parsed_data = ParsedData()
        sequence = fasta
        for i, residue in enumerate(sequence):
            if residue not in self.library:
                raise ValueError(f"Residue {residue} not found in library.")
            monomer = self.library[residue]
            parsed_monomer_info = ParsedMonomerInfo(m_idx=i, abbr=residue, res=monomer)
            parsed_data.monomers.append(parsed_monomer_info)
            # Add connection information to implicit_bond_map
            if i > 0:
                bond1 = ConnectionInfo(m_idx=i - 1, rg_idx=1)
                bond2 = ConnectionInfo(m_idx=i, rg_idx=0)
                parsed_data.implicit_bond_map[i - 1].append(bond1)
                parsed_data.implicit_bond_map[i - 1].append(bond2)
        return parsed_data


class FastaSerializer(Serializer):
    """
    Serializer for FASTA string.
    """

    def __init__(self, monomer_library: Library):
        super().__init__(monomer_library)

    def serialize(self, parsed_data: ParsedData, desc: str = "seq") -> str:
        """
        Serialize the parsed data to a FASTA string.
        """
        #fasta_str = f">{desc}\n"
        fasta_str =  ""
        for m in parsed_data.monomers:
            fasta_str += f"{m.abbr}"
        return fasta_str


if __name__ == "__main__":
    from rdkit import Chem

    lib = MonomerLibrary(
        "test_library", "/data0/yaosen/workspace/PepDB/pep/resources/monomers.sdf"
    )
    fasta = "ACDEFGHIKLMNPQRSTVWY"  # Example FASTA sequence
    parsed_data = FastaParser(lib).parse(fasta)
    print(parsed_data)
    mol = MolBuilder(parsed_data).build()
    print(Chem.MolToSmiles(mol))
    print(FastaSerializer(lib).serialize(parsed_data, "test"))
