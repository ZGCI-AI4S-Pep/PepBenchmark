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

"""
Script to format the monomer SDF file to be used in pyPept

From publication: pyPept: a python library to generate atomistic 2D and 3D representations of peptides
Journal of Cheminformatics, 2023

Instructions:

This script uses as input a set of monomers in SDF format with tags required
to index relevant atoms involved in the peptide bonds. For pyPept, we use
the public HELM monomer dataset available at:
    https://github.com/PistoiaHELM/HELMMonomerSets

The required tags are:
- name: Name of the monomer
- monomerType: Type of the monomer (AA, cap)
- polymerType: We require the type PEPTIDE
- symbol: Symbol to represent the monomer
- naturalAnalog: If the monomer has a natural AA analog
- label: Label of the R group
- capGroupName: Name of the R1 group (if exist)

The following tags optional but will be checked for when a monomer contains
the corresponding R-groups.
- capGroupName (#1): Name of the R2 group (if exist)
- capGroupName (#2): Name of the R3 group (if exist)
- capGroupName (#3): Name of the R4 group (if exist)

To run the script please provide the names of the input and out SDF files

Update log:
1. remove subtype and use type to distinguish between natural, non-natural, cap and linker
2. save monomers to SDF files based on their types
3. use loguru to record rather than logging
4. remove redundant symbol property


"""

_required_tags = (
    "name",
    "monomerType",
    "polymerType",
    "symbol",
    "naturalAnalog",
    "label",
    "capGroupName",
)

########################################################################################
# Authorship
########################################################################################

__credits__ = ["Rodrigo Ochoa", "J.B. Brown", "Thomas Fox", "Yaosen Min"]
__license__ = "MIT"
__version__ = "2.0"

########################################################################################
# Modules
########################################################################################

import argparse

# System libraries
from pathlib import Path
from string import ascii_uppercase as alc

from loguru import logger

# RDKit
from rdkit import Chem
from rdkit.Chem import PandasTools, SDWriter

# Constants expected in input SDF file:
_tag_capGroupR2 = "capGroupName (#1)"
_tag_capGroupR3 = "capGroupName (#2)"
_tag_capGroupR4 = "capGroupName (#3)"

########################################################################################
# Pipeline
########################################################################################


def generate_pdb_code(symbol, list_codes):
    """
    Function to assign PDB codes to the monomers
    :param symbol: Symbol of the monomers
    :param list_codes: List with the generated codes

    :return pdb_code: PDB code to assign
    """

    totalchar = alc + "0123456789"
    monomers = {
        "A": "ALA",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
        "M": "MET",
        "G": "GLY",
        "N": "ASN",
        "P": "PRO",
        "Q": "GLN",
        "R": "ARG",
        "S": "SER",
        "T": "THR",
        "V": "VAL",
        "W": "TRP",
        "Y": "TYR",
        "C": "CYS",
        "ac": "ACE",
        "Aib": "AIB",
        "am": "NH2",
        "Iva": "6ZS",
    }

    if symbol in monomers:
        pdb_code = monomers[symbol]
    else:
        new_symbol = symbol.replace("_", "")

        if len(new_symbol) >= 3:
            pdb_code = new_symbol[:3].upper()
            count = 0
            while pdb_code in list_codes:
                pdb_code = pdb_code[:2] + totalchar[count]
                count += 1
        else:
            pdb_code = new_symbol.upper()
            if len(pdb_code) == 2:
                for ch_val in totalchar:
                    new_code = pdb_code + ch_val
                    if new_code not in list_codes:
                        pdb_code = new_code
                        break

    return pdb_code


########################################################################################
def generate_monomers(input_file, output_file):
    """
    Function to generate the monomer SDF file for pyPept

    :param input_file: Name of the HELM monomer SDF file
    :param output_file: Name of the generated SDF file
    """

    # Reading input file
    sdf_file = input_file
    df_value = PandasTools.LoadSDF(sdf_file)
    if any([t not in df_value.columns for t in _required_tags]):
        raise RuntimeError(
            "Cannot find these required tags in SDF: "
            + " ".join([t for t in _required_tags if t not in df_value.columns])
            + ". Tags were: "
            + " ".join(df_value.columns)
        )

    # Creating output file
    writer = SDWriter(output_file)

    # List of generated PDB codes
    list_pdbs = []

    # Read the tags
    for idx in df_value.index:
        mol = df_value.at[idx, "ROMol"]
        name = df_value.at[idx, "name"]
        m_type = df_value.at[idx, "monomerType"]
        p_type = df_value.at[idx, "polymerType"]
        symbol = df_value.at[idx, "symbol"]
        symbol = symbol.replace("-", "_")
        natural_a = df_value.at[idx, "naturalAnalog"]

        # Atom order
        order = list(range(mol.GetNumAtoms()))
        indices = []
        if p_type == "PEPTIDE":
            for atom_val in mol.GetAtoms():
                if atom_val.GetSymbol()[0] == "R":
                    indices.append(atom_val.GetIdx())
                    order.remove(atom_val.GetIdx())
        for j in indices:
            order.append(j)
        # Renumber the atoms
        new_mol = Chem.RenumberAtoms(mol, newOrder=order)
        mol = new_mol

        # Extract the R-groups
        r1_group = df_value.at[idx, "capGroupName"]
        label1 = df_value.at[idx, "label"]
        if not r1_group:
            r1_group = None
        else:
            if label1 == "R2":
                r2_group = r1_group
                r1_group = None
        if label1 != "R2":
            r2_group = df_value.at[idx, _tag_capGroupR2]
            if not r2_group:
                r2_group = None
        if _tag_capGroupR3 in df_value.columns:
            r3_group = df_value.at[idx, _tag_capGroupR3]
        else:
            r3_group = None
        if _tag_capGroupR4 in df_value.columns:
            r4_group = df_value.at[idx, _tag_capGroupR4]
        else:
            r4_group = None

        # Extract the indices
        r_group_idx = [None, None, None, None]
        attachment_idx = [None, None, None, None]
        r_groups = [r1_group, r2_group, r3_group, r4_group]
        for i in range(len(r_groups)):
            if not r_groups[i]:
                r_groups[i] = None

        if p_type == "PEPTIDE":
            for atom_val in mol.GetAtoms():
                if atom_val.GetSymbol()[0] == "R":
                    label = int(atom_val.GetSymbol()[1])
                    root_atom = atom_val.GetNeighbors()[0]
                    r_group_idx[label - 1] = atom_val.GetIdx()
                    attachment_idx[label - 1] = root_atom.GetIdx()

            # Assign the categories required for the pyPept monomer dictionary
            nrg = sum([rg is not None for rg in r_groups])
            logger.info(f"Monomer: {name}, r_groups: {r_groups}, nrg: {nrg}")
            if m_type == "Backbone" and nrg >= 2:
                if symbol == natural_a:
                    cat_type = "natural"
                else:
                    cat_type = "non-natural"
            else:
                if m_type == "Backbone":
                    logger.warning(
                        f"Monomer {name} is not a peptide monomer but have Backbone annotation with {nrg} R groups"
                    )
                if nrg > 1:
                    cat_type = "linker"
                elif nrg == 1:
                    cat_type = "cap"
                else:
                    raise RuntimeError(
                        "Cannot find any R-group in the monomer: " + name
                    )
            # Add linker type, if #Rgroups > 1 --> linker, if #Rgroups = 1 --> cap

            # Generate a PDB code for the monomer
            pdb_code = generate_pdb_code(symbol, list_pdbs)
            list_pdbs.append(pdb_code)

            # Add the SDF tags
            mol.SetProp("m_name", name)
            mol.SetProp("m_abbr", symbol)
            mol.SetProp("m_type", cat_type)
            new_r_groups = [str(x) for x in r_groups]
            mol.SetProp("m_Rgroups", ",".join(new_r_groups))
            new_r_group_idx = [str(x) for x in r_group_idx]
            mol.SetProp("m_RgroupIdx", ",".join(new_r_group_idx))
            new_attachment_idx = [str(x) for x in attachment_idx]
            mol.SetProp("m_attachmentPointIdx", ",".join(new_attachment_idx))
            mol.SetProp("natAnalog", natural_a)
            mol.SetProp("pdbName", pdb_code)
            writer.write(mol)

    # Close the SDF writer
    writer.close()


# end of function generate_monomers()


def group_monomers_by_type(input_file):
    input_file = Path(input_file)
    df_value = PandasTools.LoadSDF(str(input_file))
    groups = df_value.groupby("m_type")

    for name, group in groups:
        output_file = input_file.parent / f"{name}_{input_file.name}"
        writer = SDWriter(output_file)
        for idx in group.index:
            mol = group.at[idx, "ROMol"]
            mol_name = group.at[idx, "m_name"]
            m_abbr = group.at[idx, "m_abbr"]
            m_type = group.at[idx, "m_type"]
            m_Rgroups = group.at[idx, "m_Rgroups"]
            m_RgroupIdx = group.at[idx, "m_RgroupIdx"]
            m_attachmentPointIdx = group.at[idx, "m_attachmentPointIdx"]
            natAnalog = group.at[idx, "natAnalog"]
            pdbName = group.at[idx, "pdbName"]

            # Add the SDF tags
            mol.SetProp("m_name", mol_name)
            mol.SetProp("m_abbr", m_abbr)
            mol.SetProp("m_type", m_type)
            mol.SetProp("m_Rgroups", m_Rgroups)
            mol.SetProp("m_RgroupIdx", m_RgroupIdx)
            mol.SetProp("m_attachmentPointIdx", m_attachmentPointIdx)
            mol.SetProp("natAnalog", natAnalog)
            mol.SetProp("pdbName", pdbName)
            writer.write(mol)

        writer.close()
        logger.info(f"Monomers of type {name} written to {output_file}")


##########################################################################
def get_inputs_parser():
    """Constructs parser for inputs."""

    parser = argparse.ArgumentParser(add_help=False)

    additional_type_group = parser.add_argument_group("Main options")
    additional_type_group.add_argument(
        "--input",
        type=str,
        metavar="filename",
        required=True,
        help="SDF file used as input to extract basic monomer information.",
    )
    additional_type_group.add_argument(
        "--output",
        type=str,
        metavar="filename",
        required=True,
        help="Name of the output SDF file used in pyPept.",
    )
    additional_type_group.add_argument(
        "--group-by-type",
        action="store_true",
        help="Group monomers by type into different SDF files.",
    )

    return parser


##########################################################################
# Main function
##########################################################################
def main():
    # Read arguments
    use_parser = argparse.ArgumentParser(
        description="""Generate monomer SDF file for pyPept""",
        epilog="The input SDF must have at least the following tags: "
        + " ".join(["'" + t + "'" for t in _required_tags]),
        parents=(get_inputs_parser(),),
    )
    args = use_parser.parse_args()

    logger.debug("Invocation arguments: %s" % args)
    # Run the main function
    logger.info("Generating a new monomer SDF file from %s" % args.input)
    generate_monomers(args.input, args.output)
    logger.info("Completed generation of %s" % args.output)
    if args.group_by_type:
        logger.info("Grouping monomers by type into different SDF files")
        group_monomers_by_type(args.output)


if __name__ == "__main__":
    main()
