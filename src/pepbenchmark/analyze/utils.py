"""Shared utility functions for the pepbenchmark.analyze module.

This module provides helper functions for sequence validation, normalization,
and plot management used across the analysis sub-modules.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import pandas as pd
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical ordering of the 20 standard amino acids used throughout analyses.
DEFAULT_AA_ORDER: List[str] = list("ACDEFGHIKLMNPQRSTVWY")

#: Set of the 20 standard (natural) amino acid one-letter codes.
NATURE_AA: frozenset[str] = frozenset(DEFAULT_AA_ORDER)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def save_or_show_plot(
    fig: plt.Figure,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Save a matplotlib figure to disk and/or display it interactively.

    Args:
        fig: The matplotlib ``Figure`` object to handle.
        save_path: File path to save the figure.  Supported formats are
            determined by the file extension (e.g. ``.png``, ``.pdf``).
            If ``None``, the figure is not saved.
        show: Whether to call ``plt.show()`` after saving.  Set to ``False``
            when generating figures in non-interactive (batch) environments.
    """
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Sequence normalization helpers
# ---------------------------------------------------------------------------


def to_seq_list(
    sequences: Union[List[str], pd.DataFrame],
    *,
    sequence_col: str = "sequence",
) -> List[str]:
    """Normalize a heterogeneous sequence input into a plain ``List[str]``.

    Accepts either a list/tuple of strings or a :class:`pandas.DataFrame`
    that contains a column with amino acid sequences.

    Args:
        sequences: Either a ``List[str]`` (or ``Tuple[str]``) of amino acid
            sequences, or a :class:`pandas.DataFrame` with a column named
            *sequence_col*.
        sequence_col: Name of the DataFrame column that holds the sequences.
            Ignored when *sequences* is already a list.  Defaults to
            ``"sequence"``.

    Returns:
        A flat list of sequence strings.  ``None`` entries are replaced with
        empty strings.

    Raises:
        ValueError: If *sequences* is a DataFrame and *sequence_col* is
            missing.
        TypeError: If *sequences* is neither a list/tuple nor a DataFrame.
    """
    if isinstance(sequences, pd.DataFrame):
        if sequence_col not in sequences.columns:
            raise ValueError(f"Column '{sequence_col}' not found in DataFrame")
        return sequences[sequence_col].dropna().astype(str).tolist()
    if isinstance(sequences, (list, tuple)):
        return ["" if s is None else str(s) for s in sequences]
    raise TypeError("`sequences` must be a List[str] or a pandas DataFrame")


def filter_to_natural(seq: str) -> str:
    """Remove non-canonical residues, keeping only the 20 standard amino acids.

    The input is converted to upper-case before filtering so that lower-case
    FASTA sequences are handled transparently.

    Args:
        seq: An amino acid sequence string, potentially containing ambiguous
            characters (e.g. ``X``, ``B``, ``Z``) or whitespace.

    Returns:
        A new string that contains only characters from the set of 20
        canonical amino acids in upper-case.  Returns an empty string if
        *seq* is empty or ``None``.

    Example:
        >>> filter_to_natural("ACDxEF")
        'ACDEF'
    """
    if not seq:
        return ""
    return "".join(ch for ch in seq.upper() if ch in NATURE_AA)


def validate_aa_order(aa_order: Sequence[str]) -> List[str]:
    """Validate and return a list of amino acid codes.

    Ensures that every entry in *aa_order* is one of the 20 canonical amino
    acid one-letter codes.

    Args:
        aa_order: An ordered sequence of single-character amino acid codes.

    Returns:
        A validated ``List[str]`` containing the same codes.

    Raises:
        ValueError: If *aa_order* is empty, or if any code is not among the
            20 standard amino acids.

    Example:
        >>> validate_aa_order(["A", "C", "D"])
        ['A', 'C', 'D']
    """
    order = list(aa_order)
    if not order:
        raise ValueError("`aa_order` must contain at least one amino acid code")
    for aa in order:
        if aa not in NATURE_AA:
            raise ValueError(f"Unknown amino acid in aa_order: '{aa}'")
    return order