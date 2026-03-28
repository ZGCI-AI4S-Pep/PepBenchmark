from pepbenchmark.analyze import (
    AcidLevelAnalyzer,
    KmerAnalyzer,
    SmilesAnalyse,
    compute_peptide_properties,
)


def test_kmer_analyzers_share_the_same_summary_api():
    sequences = ["ACDE", "ACDF", "AAAA"]

    acid_result = AcidLevelAnalyzer(sequences, k=2).compute_metrics()
    kmer_result = KmerAnalyzer(sequences, k=2).compute_metrics()

    assert acid_result.total_occurrences["AA"] == 3
    assert acid_result.to_dataframe().equals(kmer_result.to_dataframe())


def test_compute_peptide_properties_returns_expected_columns():
    result = compute_peptide_properties(["ACDE", "AAAA"])

    assert {"sequence", "length", "charge", "hydrophobicity"}.issubset(result.columns)
    assert len(result) == 2


def test_smiles_analysis_computes_descriptor_table():
    result = SmilesAnalyse(["CCO"]).compute_metrics().to_dataframe()

    assert {"smiles", "mol_weight", "logP"}.issubset(result.columns)
    assert len(result) == 1
