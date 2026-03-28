import os
import pandas as pd
from typing import Optional

def load_raw_pos_seqs(path: str, filter_length: Optional[int] = None) -> pd.DataFrame:
    """
    Load peptide sequences from a CSV file containing a 'sequence' column, returning a DataFrame of valid sequences.

    Args:
        path (str): Path to the CSV file containing peptide sequences.
        filter_length (Optional[int]): If set, filter out sequences longer than this length.

    Returns:
        pd.DataFrame: A DataFrame with a single column 'sequence' containing deduplicated, valid peptide sequences.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the 'sequence' column is missing in the file.
    """
    # Check file existence
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}\nPlease check the dataset path.")

    # Read CSV
    df = pd.read_csv(path)

    # Ensure 'sequence' column exists
    if "sequence" not in df.columns:
        raise ValueError(f"The file at {path} does not contain a 'sequence' column.")

    # Extract sequences, drop NaN, deduplicate
    seqs = pd.Series(df["sequence"]).dropna().astype(str).unique()

    # Filter sequences by maximum length if specified
    if filter_length is not None:
        seqs = [seq for seq in seqs if len(seq) <= filter_length]
    else:
        seqs = list(seqs)

    # Validate sequences: only standard amino acid one-letter codes
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    valid_seqs = [seq for seq in seqs if set(seq.upper()).issubset(valid_aa)]

    # Return as DataFrame
    return pd.DataFrame({'sequence': valid_seqs})