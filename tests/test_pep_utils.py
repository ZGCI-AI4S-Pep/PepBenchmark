import importlib.util

import pytest

from pepbenchmark.pep_utils.convert import Fasta2Smiles, Smiles2FP, Smiles2Graph


def test_smiles_converters_generate_molecules_and_fingerprints():
    smiles = Fasta2Smiles()("ACD")
    fingerprint = Smiles2FP(fp_type="Morgan", radius=2, nBits=32)(smiles)

    assert isinstance(smiles, str)
    assert len(fingerprint) == 32


def test_smiles_to_graph_handles_optional_dependencies():
    graph_dependencies_available = all(
        importlib.util.find_spec(name) is not None
        for name in ("ogb", "torch_geometric")
    )

    converter = Smiles2Graph()
    if graph_dependencies_available:
        graph = converter("CCO")
        assert hasattr(graph, "edge_index")
    else:
        with pytest.raises(ImportError):
            converter("CCO")
