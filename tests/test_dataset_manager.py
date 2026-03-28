from pathlib import Path

from pepbenchmark.dataset_manager.ppi_dataset import PPIDatasetManager
from pepbenchmark.dataset_manager.single_dataset import SinglePeptideDatasetManager


DATASET_DIR = Path(__file__).resolve().parents[1] / "PepBenchData" / "PepBenchData-50"


def test_single_dataset_manager_loads_official_features():
    manager = SinglePeptideDatasetManager(
        "ace_inhibitory",
        official_feature_names=["fasta", "label"],
        dataset_dir=str(DATASET_DIR),
    )

    assert manager.get_all_feature_names() == ["fasta", "label"]
    assert len(manager.get_feature("fasta")) == len(manager.get_feature("label"))


def test_ppi_dataset_manager_loads_features_and_resolves_split_aliases():
    manager = PPIDatasetManager(
        "PpI_ba",
        official_feature_names=["pep_fasta", "prot_fasta", "label"],
        dataset_dir=str(DATASET_DIR),
    )

    assert sorted(manager.features.keys()) == ["label", "pep_fasta", "prot_fasta"]
    assert manager.resolve_split_type("random_split") == "double_random_cold_split"
    assert "double_mmseqs_cold_split" in manager.get_available_official_splits()
