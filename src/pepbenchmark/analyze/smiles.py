"""SMILES-level molecular descriptor analysis utilities.

This module provides descriptor computation, atom-frequency summaries, and
distribution visualization for collections of SMILES strings.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


SUPPORTED_SMILES_PROPERTIES = [
    "mol_weight",
    "logP",
    "num_hbd",
    "num_hba",
    "tpsa",
    "num_rings",
    "num_rotatable_bonds",
    "fraction_csp3",
    "num_atoms",
    "num_atoms_h",
]


@dataclass
class SmilesPropertiesResult:
    """Store descriptor results for one or more SMILES strings.

    Attributes:
        data: Tabular descriptor data with one row per molecule.
    """

    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self, topn: int = 5) -> str:
        """Summarize the descriptor table.

        Args:
            topn: Maximum number of descriptor rows to preview.

        Returns:
            A concise textual summary.
        """
        if self.data.empty:
            return "No SMILES property results."

        lines = [f"Total molecules: {len(self.data)}"]
        description = self.data.drop(columns=["smiles"], errors="ignore").describe().T
        preview = description[["mean", "std"]].head(topn)
        lines.append("Property mean/std preview:\n" + str(preview))
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the descriptor table."""
        return self.data.copy()

    def to_json(self, indent: int = 2) -> str:
        """Serialize results to JSON.

        Args:
            indent: Number of spaces used for pretty printing.

        Returns:
            JSON text in record orientation.
        """
        return self.data.to_json(orient="records", indent=indent, force_ascii=False)

    def __str__(self) -> str:
        """Return the default string summary."""
        return self.summary()


class SmilesAnalyse:
    """Analyze molecular descriptors from a list of SMILES strings.

    Args:
        smiles_list: Input collection of SMILES strings.
    """

    def __init__(self, smiles_list: List[str]):
        self.smiles_list = list(smiles_list)
        self._result: Optional[SmilesPropertiesResult] = None

    def compute_metrics(
        self,
        properties: Optional[List[str]] = None,
    ) -> SmilesPropertiesResult:
        """Compute descriptors for the input molecules.

        Args:
            properties: Optional subset of descriptors to calculate.

        Returns:
            A :class:`SmilesPropertiesResult` instance.
        """
        rows = []
        selected_properties = properties or SUPPORTED_SMILES_PROPERTIES
        for smiles in self.smiles_list:
            try:
                rows.append(self._compute_single(smiles, selected_properties))
            except Exception:
                row = {"smiles": smiles}
                row.update({property_name: np.nan for property_name in selected_properties})
                rows.append(row)
        self._result = SmilesPropertiesResult(pd.DataFrame(rows))
        return self._result

    def summary(self) -> str:
        """Return a summary for the current result cache."""
        return self._result.summary() if self._result else "No result"

    def to_dataframe(self) -> pd.DataFrame:
        """Return cached results as a DataFrame, or an empty one."""
        return self._result.to_dataframe() if self._result else pd.DataFrame()

    def to_json(self, indent: int = 2) -> str:
        """Return cached results as JSON, or ``[]`` if unavailable."""
        return self._result.to_json(indent=indent) if self._result else "[]"

    def _compute_single(
        self,
        smiles: str,
        properties: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute descriptor values for one SMILES string.

        Args:
            smiles: Input SMILES string.
            properties: Optional subset of descriptors.

        Returns:
            A mapping of descriptor names to values.

        Raises:
            ValueError: If the SMILES string cannot be parsed.
        """
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        selected_properties = properties or SUPPORTED_SMILES_PROPERTIES
        result: Dict[str, Any] = {"smiles": smiles}
        for property_name in selected_properties:
            try:
                if property_name == "mol_weight":
                    result[property_name] = Descriptors.MolWt(molecule)
                elif property_name == "logP":
                    result[property_name] = Crippen.MolLogP(molecule)
                elif property_name == "num_hbd":
                    result[property_name] = Lipinski.NumHDonors(molecule)
                elif property_name == "num_hba":
                    result[property_name] = Lipinski.NumHAcceptors(molecule)
                elif property_name == "tpsa":
                    result[property_name] = rdMolDescriptors.CalcTPSA(molecule)
                elif property_name == "num_rings":
                    result[property_name] = rdMolDescriptors.CalcNumRings(molecule)
                elif property_name == "num_rotatable_bonds":
                    result[property_name] = Lipinski.NumRotatableBonds(molecule)
                elif property_name == "fraction_csp3":
                    result[property_name] = rdMolDescriptors.CalcFractionCSP3(molecule)
                elif property_name == "num_atoms":
                    result[property_name] = molecule.GetNumAtoms()
                elif property_name == "num_atoms_h":
                    result[property_name] = Chem.AddHs(molecule).GetNumAtoms()
                else:
                    raise ValueError(f"Unsupported property: {property_name}")
            except Exception:
                result[property_name] = np.nan
        return result

    def atom_frequency(
        self,
        include_h: bool = True,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Compute atom frequency tables for each molecule.

        Args:
            include_h: Whether to add explicit hydrogens before counting atoms.
            normalize: Whether to convert raw counts to fractions.

        Returns:
            A :class:`pandas.DataFrame` with one row per molecule.
        """
        rows = []
        for smiles in self.smiles_list:
            try:
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    raise ValueError(f"Invalid SMILES: {smiles}")
                if include_h:
                    molecule = Chem.AddHs(molecule)

                atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
                counts = Counter(atoms)
                if normalize and sum(counts.values()) > 0:
                    total = sum(counts.values())
                    counts = {atom: count / total for atom, count in counts.items()}

                row = {"smiles": smiles}
                row.update(counts)
                rows.append(row)
            except Exception:
                rows.append({"smiles": smiles})
        return pd.DataFrame(rows).fillna(0)

    def visualize_distribution(
        self,
        properties: Optional[List[str]] = None,
        kind: str = "hist",
        cols: int = 3,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """Visualize descriptor distributions across the molecule set.

        Args:
            properties: Optional subset of descriptors to plot.
            kind: Plot style: ``"hist"``, ``"kde"``, or ``"both"``.
            cols: Number of subplot columns.
            figsize: Optional overall figure size.
            save_path: Optional output image path.
            show: Whether to display the figure.

        Returns:
            The matplotlib figure and flattened axes array.
        """
        df = self.to_dataframe() if self._result else self.compute_metrics(properties).to_dataframe()
        descriptor_columns = [
            column
            for column in df.columns
            if column != "smiles" and pd.api.types.is_numeric_dtype(df[column])
        ]
        n_rows = (len(descriptor_columns) + cols - 1) // cols
        if figsize is None:
            figsize = (5 * cols, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, cols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()

        for index, property_name in enumerate(descriptor_columns):
            axis = axes[index]
            values = pd.to_numeric(df[property_name], errors="coerce").dropna()
            if kind in {"kde", "both"} and len(values) > 1:
                values.plot.kde(ax=axis, label=property_name)
            if kind in {"hist", "both"}:
                axis.hist(values, bins="auto", alpha=0.6)
            axis.set_title(property_name.replace("_", " ").title())
            axis.grid(True, alpha=0.3)

        for axis in axes[len(descriptor_columns) :]:
            axis.set_visible(False)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        return fig, axes


if __name__ == "__main__":
    example_smiles = ["CCO", "CCN", "c1ccccc1O"]
    analyzer = SmilesAnalyse(example_smiles)

    print("=== Metrics ===")
    result = analyzer.compute_metrics()
    print(result.summary())

    print("\n=== Atom frequency ===")
    print(analyzer.atom_frequency())

    print("\n=== Visualization ===")
    analyzer.visualize_distribution(properties=["mol_weight", "logP", "num_atoms"])
