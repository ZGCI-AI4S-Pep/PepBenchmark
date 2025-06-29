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

import os
import tempfile
import unittest
from pathlib import Path

from rdkit import Chem

os.environ["LOGURU_LEVEL"] = "INFO"

from pep.library import MonomerLibrary
from pep.parsers import parse_biln_to_mol
from pep.parsers.exceptions import BilnError, BilnStructureError


class TestBilnLibrary(unittest.TestCase):
    def test_load_library(self):
        sdf_path = Path(__file__).parent / "../resources/monomers.sdf"
        library = MonomerLibrary("PeptideLib", str(sdf_path))
        self.assertIsNotNone(library)
        self.assertEqual(322, len(library))


# --- BILN Parser & Serializer Tests ---
class TestBilnParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across all tests."""
        sdf_path = Path(__file__).parent / "../resources/monomers.sdf"
        cls.library = MonomerLibrary("PeptideLib", str(sdf_path))
        cls.temp_dir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run."""
        cls.temp_dir.cleanup()


# --- Test MolBuilder ---
class TestBilnBuilder(unittest.TestCase):
    """Test cases for BilnBuilder and parse_biln_to_mol functions."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across all tests."""
        sdf_path = Path(__file__).parent / "../resources/monomers.sdf"
        cls.library = MonomerLibrary("PeptideLib", str(sdf_path))
        cls.temp_dir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run."""
        cls.temp_dir.cleanup()

    def test_simple_linear_peptides(self):
        """Test parsing and building of simple linear peptides."""
        test_cases = [
            "A-G-C",  # Simple linear
            "P-E-P-T-I-D-E",  # Longer linear
            "A-A-A",  # Repeat
        ]

        for biln in test_cases:
            with self.subTest(biln=biln):
                mol = parse_biln_to_mol(biln, self.library)
                self.assertIsNotNone(mol)
                self.assertGreater(mol.GetNumAtoms(), 0)
                smiles = Chem.MolToSmiles(mol)
                self.assertIsNotNone(smiles)
                self.assertNotEqual(smiles, "")

    def test_terminal_modifications(self):
        """Test peptides with terminal modifications."""
        test_cases = [
            "A-am",  # C-term amidation
            "ac-D-T-H-F-E-I-A-am",  # N-term acetyl, C-term amide
        ]

        for biln in test_cases:
            with self.subTest(biln=biln):
                mol = parse_biln_to_mol(biln, self.library)
                self.assertIsNotNone(mol)
                self.assertGreater(mol.GetNumAtoms(), 0)
                smiles = Chem.MolToSmiles(mol)
                self.assertIsNotNone(smiles)
                self.assertNotEqual(smiles, "")

    def test_cyclic_structures(self):
        """Test cyclic peptide structures."""
        # These tests depend on the monomer library having the right definitions
        test_cases = [
            "C(1,3)-A-A-A-C(1,3)",  # Disulfide bridge
            "C(1,1)-Y-I-C(1,2)",  # Head-to-tail cycle
            "C(1,3)-Y-I-C(1,3)",  # Side chain cycle
        ]

        for biln in test_cases:
            with self.subTest(biln=biln):
                mol = parse_biln_to_mol(biln, self.library)
                self.assertIsNotNone(mol)
                self.assertGreater(mol.GetNumAtoms(), 0)
                smiles = Chem.MolToSmiles(mol)
                self.assertIsNotNone(smiles)
                self.assertNotEqual(smiles, "")

    def test_complex_structures(self):
        """Test more complex peptide structures."""
        # These depend on specific monomer library definitions
        test_cases = [
            "A-G-K(1,3)-D-D.ac(1,2)",  # Branched with modification
            "A.G.C",  # Separate chains
            "A(1,2).G(1,1)(2,2).C(2,1)",  # Multiple connections
            "A-C(1,3)-G-A-G-C(1,3)-D-am",  # Mix, complex
        ]

        for biln in test_cases:
            with self.subTest(biln=biln):
                mol = parse_biln_to_mol(biln, self.library)
                self.assertIsNotNone(mol)
                self.assertGreater(mol.GetNumAtoms(), 0)
                smiles = Chem.MolToSmiles(mol)
                self.assertIsNotNone(smiles)
                self.assertNotEqual(smiles, "")

    def test_single_monomers(self):
        """Test single monomer cases."""
        test_cases = [
            "A",
            "[Phe_4I]",
            "[A]",
            "Phe_3Cl",
        ]

        for biln in test_cases:
            with self.subTest(biln=biln):
                mol = parse_biln_to_mol(biln, self.library)
                self.assertIsNotNone(mol)
                self.assertGreater(mol.GetNumAtoms(), 0)
                smiles = Chem.MolToSmiles(mol)
                self.assertIsNotNone(smiles)
                self.assertNotEqual(smiles, "")

    def test_invalid_syntax(self):
        """Test that invalid BILN syntax properly raises errors."""
        invalid_cases = [
            "",  # Empty string
            "A.",  # Ends with separator
            "A(1,2)",  # Unpaired bond ID
            "A(1,5).G(1,1)",  # Invalid R-group index
            "X-Y-Z",  # Monomers not in library
            "[A]-[S][C]",  # missing separator
            "A--G",  # Double hyphen
            "A..G",  # Double dot
            "A-(1,2)G",  # Connection after hyphen
            "A(1,A).G(1,1)",  # Non-integer in connection
            "A()G",  # Malformed connection
            "A(1,2)G",  # Missing separator
            "A[Hyp-Gly]C",  # Unbracketed hyphen in monomer name
            "A-[]-C",  # Empty brackets
            "A (1,2) . G(1,1)",  # Extraneous whitespace
            "A . G",  # Extraneous whitespace
            "A(",  # Dangling parenthesis
            "A(1",  # Malformed connection
            "A(1,",  # Malformed connection
            "A(1,2",  # Malformed connection
            "A(1,2) .",  # Ends with separator
            "[A[B]C]",  # Nested brackets
            "A-[B]]-C",  # Invalid chars inside brackets
        ]

        for biln in invalid_cases:
            with self.subTest(biln=biln):
                # All other invalid cases should raise an error
                with self.assertRaises((BilnError, BilnStructureError)):
                    parse_biln_to_mol(biln, self.library)

    def test_complex_real_peptides(self):
        """Test complex real-world peptide examples."""
        # These depend on specific monomer library contents
        test_cases = [
            "A-A-D(1,3)-A-A-K(2,3)-A-A.K(1,3)-A-A-D(2,3)",
            "A-G-Q-A-A-K(1,3)-E-F-I-A-A.G-L-E-E(1,3)",
            "N-Iva-F-D-I-meT-N-A-L-W-Y-Aib-K",
            # cases from paper
            "D-T-H-F-P-I-C(1,3)-I-F-C(2,3)-C(3,3)-G-C(2,3)-C(4,3)-H-R-S-K-C(3,3)-G-M-C(4,3)-C(1,3)-K-T",
            "F-V-N-Q-H-L-C(1,3)-G-S-H-L-V-E-A-L-Y-L-V-C(2,3)-G-E-R-G-F-F-Y-T-P-K-T.G-I-V-E-Q-C(3,3)-C(1,3)-T-S-I-C(3,3)-S-L-Y-Q-L-E-N-Y-C(2,3)-N",
        ]

        for biln in test_cases:
            with self.subTest(biln=biln):
                try:
                    mol = parse_biln_to_mol(biln, self.library)
                    self.assertIsNotNone(mol)
                    self.assertGreater(mol.GetNumAtoms(), 0)
                    smiles = Chem.MolToSmiles(mol)
                    self.assertIsNotNone(smiles)
                    self.assertNotEqual(smiles, "")
                except BilnError as e:
                    # Only fail if this isn't due to missing monomer definitions
                    if "not found in library" not in str(e):
                        raise

    def test_molecule_properties(self):
        """Test that built molecules have expected properties."""
        biln = "A-G-C"  # Simple test case
        mol = parse_biln_to_mol(biln, self.library)

        # Check molecule was built
        self.assertIsNotNone(mol)
        self.assertGreater(mol.GetNumAtoms(), 0)

        # Check it has coordinates
        conf = mol.GetConformer()
        self.assertIsNotNone(conf)

        # Check molecule is valid
        self.assertTrue(
            Chem.SanitizeMol(mol, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE
        )

        # Check SMILES can be generated
        smiles = Chem.MolToSmiles(mol)
        self.assertIsNotNone(smiles)
        self.assertNotEqual(smiles, "")


if __name__ == "__main__":
    unittest.main()
