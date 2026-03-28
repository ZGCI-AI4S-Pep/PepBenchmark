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

from pepbenchmark.parser.base import Parser, Serializer
from pepbenchmark.parser.entity import ConnectionInfo, ParsedData, ParsedMonomerInfo
from pepbenchmark.parser.library import Library, MonomerLibrary


class HelmParser(Parser):
    """
    Parser for HELM notation strings for peptides.
    """

    def __init__(self, monomer_library: Library):
        super().__init__(monomer_library)

    def parse(self, helm_str: str) -> ParsedData:
        """
        Parse the input HELM string and return a ParsedData object.
        """
        result = ParsedData()

        # Remove version tag if present
        if "V2.0" in helm_str:
            helm_str = helm_str.replace("$V2.0", "")

        # Split the HELM string into sections
        sections = helm_str.split("$")

        # Process polymer section (may contain multiple polymers separated by |)
        polymer_section = sections[0]
        polymer_parts = polymer_section.split("|")

        # Process connection sections (if any)
        connection_sections = []
        for section in sections[1:]:
            if section and "," in section and "-" in section:
                connection_sections.append(section)

        # Parse polymers and monomers
        m_idx = 0  # Global monomer index
        polymer_info = {}  # Maps polymer name to (start_idx, length)

        for polymer_part in polymer_parts:
            if "{" not in polymer_part or "}" not in polymer_part:
                continue

            polymer_name, monomer_sequence = polymer_part.split("{", 1)
            monomer_sequence = monomer_sequence.rstrip("}")

            # Record the starting index for this polymer
            start_idx = m_idx

            # Process monomers in this polymer
            monomers = monomer_sequence.split(".")
            # New: Maintain bond_id for each polymer
            bond_id = 0
            for monomer_abbr in monomers:
                # Handle bracketed monomers
                if monomer_abbr.startswith("[") and monomer_abbr.endswith("]"):
                    clean_abbr = monomer_abbr[1:-1]
                else:
                    clean_abbr = monomer_abbr                # Look up the monomer in the library
                if clean_abbr not in self.library:
                    raise ValueError(f"Monomer '{clean_abbr}' not found in library")

                residue = self.library[clean_abbr]

                # Add the monomer to the parsed data
                monomer_info = ParsedMonomerInfo(
                    m_idx=m_idx, abbr=clean_abbr, res=residue
                )
                result.monomers.append(monomer_info)

                # Create implicit bonds between sequential monomers in the same polymer
                if m_idx > start_idx:
                    # Connect C-terminus of previous to N-terminus of current
                    prev_monomer_idx = m_idx - 1
                    result.implicit_bond_map[bond_id] = [
                        ConnectionInfo(m_idx=prev_monomer_idx, rg_idx=1),  # C-terminus
                        ConnectionInfo(m_idx=m_idx, rg_idx=0),  # N-terminus
                    ]
                    bond_id += 1

                m_idx += 1

            # Store polymer information for connection processing
            polymer_info[polymer_name] = (start_idx, len(monomers))

        # Process explicit connections
        bond_id = 0  # Assign a unique key ID for each explicit connection
        for connection_section in connection_sections:
            parts = connection_section.split(",")
            if len(parts) < 3:
                continue

            polymer1_name = parts[0]
            polymer2_name = parts[1]

            if polymer1_name not in polymer_info or polymer2_name not in polymer_info:
                continue

            polymer1_start, _ = polymer_info[polymer1_name]
            polymer2_start, _ = polymer_info[polymer2_name]

            # Process connection specifications
            for i in range(2, len(parts)):
                connection_spec = parts[i]
                if "-" not in connection_spec:
                    continue

                source_spec, target_spec = connection_spec.split("-")

                # Parse source and target positions and R-groups
                source_pos, source_rgroup = source_spec.split(":")
                target_pos, target_rgroup = target_spec.split(":")

                # Calculate monomer indices
                source_idx = polymer1_start + int(source_pos) - 1
                target_idx = polymer2_start + int(target_pos) - 1

                # Convert R-group notation to 0-based index
                source_rg_idx = int(source_rgroup.strip("R")) - 1
                target_rg_idx = int(target_rgroup.strip("R")) - 1

                # Add explicit connections with proper bond_id
                result.explicit_bond_map[bond_id] = [
                    ConnectionInfo(m_idx=source_idx, rg_idx=source_rg_idx),
                    ConnectionInfo(m_idx=target_idx, rg_idx=target_rg_idx)
                ]
                bond_id += 1

        return result


class HelmSerializer(Serializer):
    """
    Serializer for HELM notation strings for peptides.
    """

    def __init__(self, library: Library):
        super().__init__(library)

    def serialize(self, parsed_data: ParsedData) -> str:
        """
        Serialize the ParsedData object to a HELM string.

        If the ParsedData was originally created from a HELM string,
        this will try to preserve the original format.
        """
        # Otherwise, build a new HELM string
        if not parsed_data.monomers:
            return "$$$V2.0"  # Empty HELM string

        # Group monomers into polymers based on implicit bonds
        polymers = []
        current_polymer = []
        processed_indices = set()

        # First, identify linear chains using implicit bonds
        for monomer in sorted(parsed_data.monomers, key=lambda m: m.m_idx):
            if monomer.m_idx in processed_indices:
                continue

            # Start a new polymer chain
            current_polymer = [monomer]
            processed_indices.add(monomer.m_idx)

            # Follow implicit bonds to build the chain
            current_idx = monomer.m_idx
            while True:
                # Find the next monomer in the chain via implicit bonds
                next_monomer = None
                for conn in parsed_data.implicit_bond_map.get(current_idx, []):
                    # Check if this is a peptide bond (C-terminus to N-terminus)
                    if conn.rg_idx == 0:  # N-terminus connection
                        next_idx = conn.m_idx
                        if next_idx not in processed_indices:
                            next_monomer = next(
                                (
                                    m
                                    for m in parsed_data.monomers
                                    if m.m_idx == next_idx
                                ),
                                None,
                            )
                            if next_monomer:
                                break

                if next_monomer:
                    current_polymer.append(next_monomer)
                    processed_indices.add(next_monomer.m_idx)
                    current_idx = next_monomer.m_idx
                else:
                    break

            if current_polymer:
                polymers.append(current_polymer)

        # Handle any remaining monomers as individual polymers
        for monomer in parsed_data.monomers:
            if monomer.m_idx not in processed_indices:
                polymers.append([monomer])
                processed_indices.add(monomer.m_idx)

        # Build the polymer section of the HELM string
        polymer_strings = []
        for i, polymer in enumerate(polymers):
            # Format monomers with brackets for non-natural amino acids if needed
            monomer_strings = []
            for m in polymer:
                if m.res.m_type.value == "non-natural" and not (
                    m.abbr.startswith("[") and m.abbr.endswith("]")
                ):
                    monomer_strings.append(f"[{m.abbr}]")
                else:
                    monomer_strings.append(m.abbr)

            monomer_string = ".".join(monomer_strings)
            polymer_strings.append(f"PEPTIDE{i+1}{{{monomer_string}}}")

        polymer_section = "|".join(polymer_strings)

        # Create a map of monomer indices to their polymer and position
        monomer_locations = {}
        for polymer_idx, polymer in enumerate(polymers):
            for pos_idx, monomer in enumerate(polymer):
                monomer_locations[monomer.m_idx] = (polymer_idx + 1, pos_idx + 1)

        # Add debugging information to help validate connection processing
        if parsed_data.explicit_bond_map:
            print(f"DEBUG: Processing {len(parsed_data.explicit_bond_map)} explicit bonds")
            for bond_id, connections in parsed_data.explicit_bond_map.items():
                print(f"DEBUG: Bond {bond_id}: {len(connections)} connections")
                for conn in connections:
                    print(f"DEBUG:   - Monomer {conn.m_idx}, R-group {conn.rg_idx}")
        else:
            print("DEBUG: No explicit bonds found in parsed data - this is normal for linear peptides")

        # Extract explicit connections
        connections = {}  # Maps (polymer1, polymer2) to list of connection specs

        # Process explicit bonds - fix connection processing logic
        processed_bonds = set()

        # Redesign connection processing logic to ensure correct pairing
        for bond_id, bond_connections in parsed_data.explicit_bond_map.items():
            # Each bond_id should have two connections (source and target)
            if len(bond_connections) != 2:
                continue
                
            if bond_id in processed_bonds:
                continue
                
            processed_bonds.add(bond_id)
            
            # Get the two endpoints of the connection
            conn1, conn2 = bond_connections[0], bond_connections[1]
            
            # Skip if location information is incomplete
            if (conn1.m_idx not in monomer_locations or 
                conn2.m_idx not in monomer_locations):
                continue
                
            source_polymer, source_pos = monomer_locations[conn1.m_idx]
            target_polymer, target_pos = monomer_locations[conn2.m_idx]
            
            # Create connection specification - fix R-group index
            # Note: BILN uses 1-based indexing, and HELM also uses 1-based indexing
            conn_spec = f"{source_pos}:R{conn1.rg_idx + 1}-{target_pos}:R{conn2.rg_idx + 1}"
            
            # Handle connections within the same polymer (cyclization) or between different polymers
            if source_polymer == target_polymer:
                polymer_key = (f"PEPTIDE{source_polymer}", f"PEPTIDE{source_polymer}")
            else:
                polymer_key = (f"PEPTIDE{source_polymer}", f"PEPTIDE{target_polymer}")
                
            if polymer_key not in connections:
                connections[polymer_key] = []
            connections[polymer_key].append(conn_spec)

        # Add connection processing validation and debugging information
        if connections:
            print(f"DEBUG: Successfully processed {len(connections)} connection groups")
            for polymer_key, conn_specs in connections.items():
                print(f"DEBUG: {polymer_key}: {conn_specs}")
        else:
            if parsed_data.explicit_bond_map:
                print("DEBUG: WARNING: No connections were processed despite having explicit bonds - this may indicate an issue")
            else:
                print("DEBUG: No connections to process - this is normal for linear peptides")

        # Build the connection section
        connection_strings = []

        for (polymer1, polymer2), conn_specs in connections.items():
            # For each polymer pair, create a connection string
            for spec in conn_specs:
                connection_strings.append(f"{polymer1},{polymer2},{spec}")

        # Combine all sections
        if connection_strings:
            connection_section = "$" + "$".join(connection_strings)
        else:
            connection_section = "$"

        return f"{polymer_section}{connection_section}$$V2.0"


if __name__ == "__main__":
    from pepbenchmark.parser.builder import MolBuilder
    from rdkit import Chem

    helm_str = "PEPTIDE1{A.G.C}$$$V2.0"

    helm_list = [
        "PEPTIDE1{A.[meA].C}$$$$",
        "PEPTIDE1{D.E.F.G}|PEPTIDE2{C.E}$PEPTIDE1,PEPTIDE2,2:R3-1:R1$$$V2.0",
        "PEPTIDE1{L.M.P.Q.R.S.T}$PEPTIDE1,PEPTIDE1,7:R2-1:R1$$$",
        "PEPTIDE1{N.P.F.V.L.P.[dV]}$PEPTIDE1,PEPTIDE1,7:R2-1:R1$$$",
        "PEPTIDE1{A.R.C.A.A.K.T.C.D.A}$PEPTIDE1,PEPTIDE1,8:R3-3:R3$$$",
    ]

    lib = MonomerLibrary.from_sdf_file(
        "test_library", "/home/dataset-assist-0/rouyi/rouyi/Projects/PepBenchmark/src/pepbenchmark/external/pep/resources/monomers.sdf"
    )
    for helm in helm_list:
        parser = HelmParser(lib)
        parsed_data = parser.parse(helm)
        print(parsed_data)
        serializer = HelmSerializer(lib)
        serialized_helm = serializer.serialize(parsed_data)
        print(helm)
        print(serialized_helm)
        mol = MolBuilder(parsed_data).build()
        smiles = Chem.MolToSmiles(mol)
        print(smiles)
        print("==================")
#
