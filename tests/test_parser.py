from pathlib import Path

from pepbenchmark.parser.fasta_parser import FastaParser, FastaSerializer
from pepbenchmark.parser.library import ConcatLibrary, MonomerLibrary


MONOMER_SDF = Path(__file__).resolve().parents[1] / "src" / "pepbenchmark" / "parser" / "monomers_merged.sdf"


def test_fasta_parser_roundtrip_uses_packaged_monomer_library():
    library = MonomerLibrary.from_sdf_file("default", MONOMER_SDF)
    parsed = FastaParser(library).parse("ACD")

    assert len(parsed.monomers) == 3
    assert FastaSerializer(library).serialize(parsed) == "ACD"


def test_concat_library_supports_membership_and_integer_indexing():
    library = MonomerLibrary.from_sdf_file("default", MONOMER_SDF)
    lib_a = MonomerLibrary.from_monomer_list("a", [library["A"], library["C"]])
    lib_b = MonomerLibrary.from_monomer_list("b", [library["D"]])
    merged = ConcatLibrary("merged", [lib_a, lib_b])

    assert "A" in merged
    assert merged[0].m_abbr == "A"
    assert merged[2].m_abbr == "D"
