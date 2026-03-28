"""Public exports for peptide and molecular analysis helpers.

Some analysis submodules depend on optional plotting or chemistry libraries.
This package keeps lightweight functionality importable even when those
optional dependencies are unavailable.
"""

from pepbenchmark.analyze.fasta_level import (
	PeptidePropertiesAnalyse,
	PeptidePropertiesResult,
	PropertyComparisonResult,
	ValidationOptions,
	compute_peptide_properties,
)

__all__ = [
	"PeptidePropertiesAnalyse",
	"PeptidePropertiesResult",
	"PropertyComparisonResult",
	"ValidationOptions",
	"compute_peptide_properties",
]

from pepbenchmark.analyze.acid_level import AcidLevelAnalyzer, KmerAnalyse as AcidKmerAnalyse, KmerStats as AcidKmerStats

__all__.extend([
	"AcidKmerAnalyse",
	"AcidKmerStats",
	"AcidLevelAnalyzer",
])

from pepbenchmark.analyze.kmer_level import KmerAnalyzer, KmerAnalyse, KmerStats

__all__.extend([
	"KmerAnalyse",
	"KmerAnalyzer",
	"KmerStats",
])

from pepbenchmark.analyze.smiles import SmilesAnalyse, SmilesPropertiesResult

__all__.extend([
	"SmilesAnalyse",
	"SmilesPropertiesResult",
])
