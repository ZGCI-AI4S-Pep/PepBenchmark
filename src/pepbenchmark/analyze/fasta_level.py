"""FASTA-level peptide property analysis utilities.

This module computes physicochemical properties from peptide sequences and
supports batch comparison and quick visual inspection of the resulting
distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


AA_STD: Tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_STD)
AMBIGUOUS = set("XBZJUO")


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError("Visualization requires optional dependency: matplotlib")


@dataclass(frozen=True)
class ValidationOptions:
    """Configuration for sequence validation.

    Attributes:
        allow_ambiguous: Whether to allow ambiguous residue codes.
        uppercase: Whether to uppercase sequences before validation.
    """

    allow_ambiguous: bool = False
    uppercase: bool = True


def _normalize_sequence(seq: str, opts: ValidationOptions) -> str:
    """Normalize and validate a peptide sequence.

    Args:
        seq: Raw sequence string.
        opts: Validation behavior.

    Returns:
        The validated, normalized sequence.

    Raises:
        ValueError: If the sequence is empty, missing, or contains invalid
            residues.
    """
    if seq is None:
        raise ValueError("Sequence is None")

    normalized = str(seq).strip()
    if opts.uppercase:
        normalized = normalized.upper()
    if not normalized:
        raise ValueError("Empty sequence")

    unknown = set(normalized) - AA_SET
    if unknown and not (opts.allow_ambiguous and unknown <= AMBIGUOUS):
        raise ValueError(f"Invalid amino acids: {sorted(unknown)}")
    return normalized


@lru_cache(maxsize=4096)
def _analyze(seq: str) -> ProteinAnalysis:
    """Cache ``ProteinAnalysis`` objects for repeated computations."""
    return ProteinAnalysis(seq)


@dataclass
class PeptidePropertiesResult:
    """Store computed peptide-property results.

    Attributes:
        data: Tabular property data with one row per sequence.
    """

    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self, topn: int = 5) -> str:
        """Summarize the computed properties.

        Args:
            topn: Maximum number of numeric properties to summarize.

        Returns:
            A concise text summary.
        """
        if self.data.empty:
            return "No properties computed."

        lines = [f"Total sequences: {len(self.data)}"]
        numeric_columns = [
            column
            for column in self.data.columns
            if column != "sequence" and pd.api.types.is_numeric_dtype(self.data[column])
        ]
        for column in numeric_columns[:topn]:
            lines.append(
                f"{column}: mean={self.data[column].mean():.3f}, "
                f"std={self.data[column].std():.3f}"
            )
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the underlying result table."""
        return self.data.copy()

    def __str__(self) -> str:
        """Return the default string summary."""
        return self.summary()


@dataclass
class PropertyComparisonResult:
    """Store pairwise distribution comparison results.

    Attributes:
        data: Comparison statistics with one row per property.
    """

    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        """Create a text summary of comparison metrics."""
        if self.data.empty:
            return "No comparison results."
        return "\n".join(
            (
                f"{row['property']}: KS p={row['ks_pvalue']:.3f}, "
                f"mean_diff={row['mean_diff']:.3f}, JS={row['js_divergence']:.3f}"
            )
            for _, row in self.data.iterrows()
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the comparison table."""
        return self.data.copy()

    def __str__(self) -> str:
        """Return the default string summary."""
        return self.summary()


class PeptidePropertiesAnalyse:
    """Compute peptide physicochemical properties from FASTA-like sequences.

    Args:
        ph: pH value used for charge estimation.
        validate: Whether to validate sequences before analysis.
        allow_ambiguous: Whether to allow ambiguous amino acid codes.
    """

    def __init__(
        self,
        ph: float = 7.0,
        validate: bool = True,
        allow_ambiguous: bool = False,
    ) -> None:
        self.ph = ph
        self.validate = validate
        self.allow_ambiguous = allow_ambiguous

    def _property_functions(self) -> Dict[str, Callable[[ProteinAnalysis, str], Any]]:
        """Build the property registry used during computation."""

        def _flex_mean(analysis: ProteinAnalysis, _sequence: str) -> float:
            values = analysis.flexibility()
            return float(np.mean(values)) if len(values) else 0.0

        def _secondary_structure_fraction(
            analysis: ProteinAnalysis,
            _sequence: str,
            index: int,
        ) -> float:
            helix, turn, sheet = analysis.secondary_structure_fraction()
            return float((helix, turn, sheet)[index])

        def _aliphatic_index(analysis: ProteinAnalysis, _sequence: str) -> float:
            counts = analysis.count_amino_acids()
            total = sum(counts.values())
            if total == 0:
                return 0.0
            alanine = counts.get("A", 0)
            valine = counts.get("V", 0)
            isoleucine = counts.get("I", 0)
            leucine = counts.get("L", 0)
            return float((alanine + 2.9 * valine + 3.9 * (isoleucine + leucine)) / total * 100)

        return {
            "length": lambda analysis, sequence: len(sequence),
            "mw": lambda analysis, sequence: float(analysis.molecular_weight()),
            "aromaticity": lambda analysis, sequence: float(analysis.aromaticity()),
            "instability": lambda analysis, sequence: float(analysis.instability_index()),
            "isoelectric_point": lambda analysis, sequence: float(analysis.isoelectric_point()),
            "helix": lambda analysis, sequence: _secondary_structure_fraction(analysis, sequence, 0),
            "turn": lambda analysis, sequence: _secondary_structure_fraction(analysis, sequence, 1),
            "sheet": lambda analysis, sequence: _secondary_structure_fraction(analysis, sequence, 2),
            "hydrophobicity": lambda analysis, sequence: float(analysis.gravy()),
            "charge": lambda analysis, sequence: float(analysis.charge_at_pH(self.ph)),
            "charge_at_ph7": lambda analysis, sequence: float(analysis.charge_at_pH(7.0)),
            "flexibility": _flex_mean,
            "aliphatic_index": _aliphatic_index,
        }

    def _normalize_many(
        self,
        data: Union[pd.DataFrame, Sequence[str], str, Iterable[str]],
        seq_col: str = "sequence",
    ) -> List[str]:
        """Normalize heterogeneous input into a validated list of sequences."""
        if isinstance(data, pd.DataFrame):
            sequences = data[seq_col].dropna().astype(str).tolist()
        elif isinstance(data, str):
            sequences = [data]
        else:
            sequences = list(data)

        if self.validate:
            options = ValidationOptions(allow_ambiguous=self.allow_ambiguous)
            return [_normalize_sequence(sequence, options) for sequence in sequences]
        return [str(sequence).upper().strip() for sequence in sequences]

    def compute(
        self,
        data: Union[pd.DataFrame, Sequence[str], str, Iterable[str]],
        properties: Optional[List[str]] = None,
        seq_col: str = "sequence",
    ) -> PeptidePropertiesResult:
        """Compute physicochemical properties for one or more sequences.

        Args:
            data: Input sequence collection.
            properties: Optional subset of properties to compute.
            seq_col: Sequence column name when *data* is a DataFrame.

        Returns:
            A :class:`PeptidePropertiesResult` instance.
        """
        sequences = self._normalize_many(data, seq_col=seq_col)
        property_functions = self._property_functions()

        selected_properties = properties or list(property_functions.keys())
        rows: List[Dict[str, Any]] = []
        for sequence in sequences:
            analysis = _analyze(sequence)
            row: Dict[str, Any] = {"sequence": sequence}
            for property_name in selected_properties:
                try:
                    row[property_name] = property_functions[property_name](analysis, sequence)
                except Exception:
                    row[property_name] = np.nan
            rows.append(row)
        return PeptidePropertiesResult(data=pd.DataFrame(rows))

    def compare(
        self,
        data1: Union[pd.DataFrame, Sequence[str], str, Iterable[str]],
        data2: Union[pd.DataFrame, Sequence[str], str, Iterable[str]],
        properties: Optional[List[str]] = None,
        seq_col: str = "sequence",
    ) -> PropertyComparisonResult:
        """Compare property distributions between two sequence groups.

        Args:
            data1: First sequence collection.
            data2: Second sequence collection.
            properties: Optional subset of properties to compare.
            seq_col: Sequence column name when either input is a DataFrame.

        Returns:
            A :class:`PropertyComparisonResult` instance.
        """
        df1 = self.compute(data1, properties, seq_col=seq_col).to_dataframe()
        df2 = self.compute(data2, properties, seq_col=seq_col).to_dataframe()

        selected_properties = properties or [column for column in df1.columns if column != "sequence"]
        output_rows: List[Dict[str, float | int | str]] = []
        for property_name in selected_properties:
            values_1 = df1[property_name].dropna().to_numpy()
            values_2 = df2[property_name].dropna().to_numpy()
            if len(values_1) == 0 or len(values_2) == 0:
                continue

            ks_stat, ks_pvalue = ks_2samp(values_1, values_2)
            mean_diff = float(np.mean(values_1) - np.mean(values_2))
            std_diff = (
                float(np.std(values_1, ddof=1) - np.std(values_2, ddof=1))
                if len(values_1) > 1 and len(values_2) > 1
                else np.nan
            )

            try:
                lower_bound = min(values_1.min(), values_2.min())
                upper_bound = max(values_1.max(), values_2.max())
                bins = np.linspace(lower_bound, upper_bound, 21)
                hist_1, _ = np.histogram(values_1, bins=bins)
                hist_2, _ = np.histogram(values_2, bins=bins)
                prob_1 = hist_1 / hist_1.sum() if hist_1.sum() > 0 else None
                prob_2 = hist_2 / hist_2.sum() if hist_2.sum() > 0 else None
                js_divergence = (
                    float(jensenshannon(prob_1, prob_2))
                    if prob_1 is not None and prob_2 is not None
                    else np.nan
                )
            except Exception:
                js_divergence = np.nan

            output_rows.append(
                {
                    "property": property_name,
                    "ks_stat": float(ks_stat),
                    "ks_pvalue": float(ks_pvalue),
                    "mean_diff": mean_diff,
                    "std_diff": std_diff,
                    "js_divergence": js_divergence,
                    "n_group1": len(values_1),
                    "n_group2": len(values_2),
                }
            )
        return PropertyComparisonResult(data=pd.DataFrame(output_rows))

    def visualize(
        self,
        data: Union[pd.DataFrame, Sequence[str], str, Iterable[str]],
        properties: Optional[List[str]] = None,
        seq_col: str = "sequence",
        kind: str = "hist",
        cols: int = 3,
        figsize: Optional[Tuple[int, int]] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ):
        """Visualize computed property distributions.

        Args:
            data: Input sequence collection.
            properties: Optional subset of properties to plot.
            seq_col: Sequence column name when *data* is a DataFrame.
            kind: Plot style: ``"hist"``, ``"kde"``, or ``"both"``.
            cols: Number of subplot columns.
            figsize: Optional overall figure size.
            show: Whether to display the plot.
            save_path: Optional path to save the figure.

        Returns:
            The matplotlib figure and flattened axes array.
        """
        _require_matplotlib()
        df = self.compute(data, properties, seq_col).to_dataframe()
        selected_properties = [column for column in df.columns if column != "sequence"]
        n_rows = (len(selected_properties) + cols - 1) // cols
        if figsize is None:
            figsize = (5 * cols, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, cols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()

        for index, property_name in enumerate(selected_properties):
            axis = axes[index]
            values = df[property_name].dropna()
            if kind in {"kde", "both"} and len(values) > 1:
                values.plot.kde(ax=axis)
            if kind in {"hist", "both"}:
                axis.hist(values, bins="auto", alpha=0.6)
            axis.set_title(property_name)

        for axis in axes[len(selected_properties) :]:
            axis.set_visible(False)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig, axes


def compute_peptide_properties(
    seqs: List[str],
    properties: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute peptide properties and return them as a DataFrame.

    Args:
        seqs: List of peptide sequences.
        properties: Optional subset of properties to compute.

    Returns:
        A :class:`pandas.DataFrame` with one row per sequence.
    """
    analyzer = PeptidePropertiesAnalyse(ph=7.0)
    return analyzer.compute(seqs, properties).to_dataframe()


if __name__ == "__main__":
    analyzer = PeptidePropertiesAnalyse(ph=7.0)

    example_sequences = ["ACDEFGHIK", "LMNPQRST", "MNPQRSTVWY"]
    print(compute_peptide_properties(example_sequences))

    result = analyzer.compute(example_sequences)
    print(result)

    comparison = analyzer.compare(
        example_sequences[:2],
        example_sequences[2:],
        properties=["length", "charge"],
    )
    print(comparison)

    analyzer.visualize(
        example_sequences,
        properties=["length", "charge", "hydrophobicity"],
        kind="both",
    )
