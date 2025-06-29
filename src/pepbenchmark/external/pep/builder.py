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
from typing import List

from loguru import logger
from pep.entity import ParsedData
from rdkit import Chem
from rdkit.Chem import AllChem


class MolBuilderError(ValueError):
    """Error during molecule construction (missing monomers, bad R-groups, etc.)."""

    pass


class MolBuilder:
    """Builds an RDKit Mol object from parsed data object."""

    LBL_ATOM_RM = "____label_atom_rm"
    LBL_BOND_ADD = "____label_bond_add"
    LBL_GRP_ADD = "____label_group_add"

    def __init__(self, parsed_data: ParsedData):
        self.data = parsed_data
        self.combined_mol = Chem.RWMol()
        self.atom_offsets: List[int] = [0] * len(self.data.monomers)
        self._bond_serial = 0

    @property
    def bond_serial(self) -> int:
        """Returns the current bond serial number and increments it."""
        self._bond_serial += 1
        return self._bond_serial

    def build(self) -> Chem.Mol:
        """Constructs the final RDKit Mol."""
        self._assemble_initial_fragments()
        self._plan_bonds_and_removals()
        self._plan_recover_other_rgroups()
        # Atom index in self.combined_mol is changed! Should not use self.data anymore
        self._execute_bonds_and_removals()
        self._execute_recover_other_rgroups()
        return self._finalize_molecule()

    def _assemble_initial_fragments(self):
        """Adds all monomer fragments to the combined RWMol."""
        combined_mol = self.data.monomers[0].res.m_romol
        self.atom_offsets[0] = 0
        # TODO: How to ensure the order of self.data.monomers and m_info.instance_id have the same value?
        for idx, m_info in enumerate(self.data.monomers[1:], 1):
            assert (
                m_info.m_idx == idx
            ), f"Monomer instance ID mismatch: expected {idx}, got {m_info.m_idx}"
            self.atom_offsets[m_info.m_idx] = combined_mol.GetNumAtoms()
            combined_mol = Chem.CombineMols(combined_mol, m_info.res.m_romol)

        self.combined_mol = Chem.RWMol(combined_mol)  # Convert to RWMol for editing

    def _mark_atom_deletion(self, atom_idx: int):
        """Mark atoms for later ops"""
        atom = self.combined_mol.GetAtomWithIdx(atom_idx)
        atom.SetIntProp(MolBuilder.LBL_ATOM_RM, 1)

    def _mark_bond_addition(self, atom_idx1: int, atom_idx2: int, serial_num: int):
        """Mark bonds for later ops"""
        atom1 = self.combined_mol.GetAtomWithIdx(atom_idx1)
        atom2 = self.combined_mol.GetAtomWithIdx(atom_idx2)
        atom1.SetIntProp(MolBuilder.LBL_BOND_ADD, serial_num)
        atom2.SetIntProp(MolBuilder.LBL_BOND_ADD, serial_num)

    def _mark_addition_at_atom(self, atom_idx: int, group_smi: str):
        atom = self.combined_mol.GetAtomWithIdx(atom_idx)
        atom.SetProp(MolBuilder.LBL_GRP_ADD, group_smi)

    def _plan_bonds_and_removals(self):
        """Determines which bonds to add and atoms to remove based on BILN."""
        for bond_id, connections in self.data.implicit_bond_map.items():
            c1 = connections[0]
            c2 = connections[1]
            logger.debug(
                f"Planning implicit peptide bond {bond_id}: {self.data.monomers[c1.m_idx].abbr}(R{c1.rg_idx+1}) - {self.data.monomers[c2.m_idx].abbr}(R{c2.rg_idx+1})"
            )
            self._plan_single_bond(c1.m_idx, c1.rg_idx, c2.m_idx, c2.rg_idx)

        for bond_id, connections in self.data.explicit_bond_map.items():
            c1 = connections[0]
            c2 = connections[1]
            logger.debug(
                f"Planning explicit bond {bond_id}: {self.data.monomers[c1.m_idx].abbr}(R{c1.rg_idx+1}) - {self.data.monomers[c2.m_idx].abbr}(R{c2.rg_idx+1})"
            )
            self._plan_single_bond(c1.m_idx, c1.rg_idx, c2.m_idx, c2.rg_idx)

    def _combined_index(self, inst_id: int, ori_idx: int) -> int:
        return self.atom_offsets[inst_id] + ori_idx

    def _plan_single_bond(self, inst_id1: int, r_idx1: int, inst_id2: int, r_idx2: int):
        """Plans one bond addition and associated atom removals."""
        # in _assemble_initial_fragments, monomers are definitely combined in order using rdkit
        # See https://github.com/rdkit/rdkit/blob/b3076c77284b9a8b9d5ef78957ee067037f373a8/Code/GraphMol/ChemTransforms/ChemTransforms.cpp#L866
        # See https://github.com/rdkit/rdkit/blob/b3076c77284b9a8b9d5ef78957ee067037f373a8/Code/GraphMol/RWMol.cpp#L142

        # Get R-group info (content, leaving_idx, attach_idx)
        res1 = self.data.monomers[inst_id1].res
        res2 = self.data.monomers[inst_id2].res

        rinfo1 = res1.m_Rgroups[r_idx1]
        rinfo2 = res2.m_Rgroups[r_idx2]

        leaving_orig_idx1 = rinfo1[1]
        attach_orig_idx1 = rinfo1[2]
        leaving_orig_idx2 = rinfo2[1]
        attach_orig_idx2 = rinfo2[2]
        # --- Get corresponding indices in the combined molecule ---
        attach_comb_idx1 = self._combined_index(inst_id1, attach_orig_idx1)
        attach_comb_idx2 = self._combined_index(inst_id2, attach_orig_idx2)
        # --- Plan bond addition ---
        # Store the attachment indices *before* removal
        self._mark_bond_addition(attach_comb_idx1, attach_comb_idx2, self.bond_serial)
        # --- Plan atom removals ---
        # Remove the 'leaving' atom specified in the R-group tuple
        leaving_comb_idx1 = self._combined_index(inst_id1, leaving_orig_idx1)
        self._mark_atom_deletion(leaving_comb_idx1)
        leaving_comb_idx2 = self._combined_index(inst_id2, leaving_orig_idx2)
        self._mark_atom_deletion(leaving_comb_idx2)

    def _plan_recover_other_rgroups(self):
        """Recover R groups by the definition if it is not used in self._plan_bonds_and_removals()"""
        # iterate all R groups and check atoms at leaving_comb_idx
        # if it is marked for deletion, skip
        # then we just add the corresponding rgroups, connect bond, mark atom for deletion
        atom_marked_rm = set()
        for atom in self.combined_mol.GetAtoms():
            if atom.HasProp(MolBuilder.LBL_ATOM_RM):
                atom_marked_rm.add(atom.GetIdx())

        for m_idx, m_info in enumerate(self.data.monomers):
            for r_idx, rgroup in enumerate(m_info.res.m_Rgroups):
                # rgroup = (content_smi, leaving_orig_idx, attach_orig_idx)
                if rgroup[0] is None:
                    continue
                rgoup_smi, leaving_orig_idx, attach_orig_idx = rgroup
                leaving_comb_idx = self._combined_index(m_idx, leaving_orig_idx)
                attach_comb_idx = self._combined_index(m_idx, attach_orig_idx)
                if leaving_comb_idx in atom_marked_rm:
                    # This atom is marked for deletion, must got something to bond to at atatch_orig_idx
                    continue
                logger.debug(
                    f"Recovering {m_idx}-th monomer {m_info.abbr}, R-group {r_idx+1} with '{rgoup_smi}'"
                )
                self._mark_atom_deletion(leaving_comb_idx)
                self._mark_addition_at_atom(attach_comb_idx, rgoup_smi)

    def _execute_bonds_and_removals(self):
        """Removes planned atoms from combined_mol, handling index changes."""
        atoms_to_remove = list()
        for atom in self.combined_mol.GetAtoms():
            if atom.HasProp(MolBuilder.LBL_ATOM_RM):
                atom_idx = atom.GetIdx()
                atoms_to_remove.append(atom_idx)

        # IMPORTANT: Remove atoms in descending order of index
        # This ensures that removing an atom doesn't affect the index of atoms
        # yet to be removed in this loop.
        atoms_to_remove = sorted(atoms_to_remove, reverse=True)
        for atom_idx in atoms_to_remove:
            self.combined_mol.RemoveAtom(atom_idx)

        # Add bonds
        atom_pair_add_bonds = defaultdict(list)
        for atom in self.combined_mol.GetAtoms():
            if atom.HasProp(MolBuilder.LBL_BOND_ADD):
                serial_num = atom.GetIntProp(MolBuilder.LBL_BOND_ADD)
                atom_pair_add_bonds[serial_num].append(
                    atom.GetIdx()
                )  # bond serials are strange? debugger???
        # Add bonds in the order of their serial numbers
        for serial_num, atom_indices in sorted(atom_pair_add_bonds.items()):
            if len(atom_indices) != 2:
                raise MolBuilderError(
                    f"Invalid bond addition: Expected exactly 2 atoms for serial {serial_num}, found {len(atom_indices)}."
                )
            self.combined_mol.AddBond(
                atom_indices[0], atom_indices[1], Chem.BondType.SINGLE
            )

    def _execute_recover_other_rgroups(self):
        """Adds R-groups to the combined molecule."""
        ori_num_atoms = self.combined_mol.GetNumAtoms()
        for atom_idx in range(ori_num_atoms):
            atom = self.combined_mol.GetAtomWithIdx(atom_idx)
            if atom.HasProp(MolBuilder.LBL_GRP_ADD):
                group_smi = atom.GetProp(MolBuilder.LBL_GRP_ADD)
                # Add the R-group as a new fragment
                last_atom_idx = self.combined_mol.GetNumAtoms() - 1
                if group_smi == "OH":
                    group_mol = Chem.MolFromSmiles(group_smi.replace("H", "[H]"))
                    self.combined_mol = Chem.RWMol(
                        Chem.CombineMols(self.combined_mol, group_mol)
                    )
                    self.combined_mol.AddBond(
                        atom_idx, last_atom_idx + 1, Chem.BondType.SINGLE
                    )
                elif group_smi == "H":
                    # just remove the R group and it will be fine
                    pass
                else:
                    raise MolBuilderError(
                        f"Unrecognized R-group SMILES '{group_smi}' at atom index {atom_idx}, current Mol is {Chem.MolToSmiles(self.combined_mol)}."
                    )

    def _finalize_molecule(self) -> Chem.Mol:
        """Converts RWMol to Mol, sanitizes, and computes coordinates."""
        for atom in self.combined_mol.GetAtoms():
            atom.ClearProp(MolBuilder.LBL_ATOM_RM)
            atom.ClearProp(MolBuilder.LBL_BOND_ADD)
            atom.ClearProp(MolBuilder.LBL_GRP_ADD)
        # Get the final molecule object
        final_mol = self.combined_mol.GetMol()
        # Clean up, sanitize, calculate coordinates
        Chem.SanitizeMol(final_mol)
        # TODO: add 3D coordinates to molecules
        AllChem.Compute2DCoords(final_mol)
        return final_mol
