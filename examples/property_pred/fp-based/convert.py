import argparse
import os
import time

import pandas as pd
from tqdm import tqdm
from utils import FP_Converter, featurize, seq_to_mol, seq_to_smiles

from pepbenchmark.metadata import DATASET_MAP

tqdm.pandas()


def process_split_files(
    base_dir: str, converter: FP_Converter, fp_type: str, nbits: int, radius: int
) -> (int, float, str):
    """
    Process combined CSV file for a dataset:
     - Reads CSV
     - Applies featurization to the sequence column with progress bars
     - Adds a SMILES column
     - Saves new CSV with fingerprint features

    Returns:
        data_count: number of records processed
        elapsed: processing time in seconds
        out_path: path where output was saved
    """
    start = time.time()

    # Read input
    base_path = os.path.join(base_dir, "combine.csv")
    df = pd.read_csv(base_path)
    data_count = len(df)

    # Featurize with progress
    tqdm.pandas(desc="Converting sequences to SMILES")
    df["SMILES"] = df["sequence"].progress_apply(seq_to_smiles)

    fps = []
    for fp in tqdm(
        featurize(df["sequence"], converter),
        total=data_count,
        desc="Generating fingerprints",
    ):
        fps.append(fp.tolist())
    df["fp"] = fps

    # Prepare output directory and save
    save_name = f"{fp_type}_nbits{nbits}_radius{radius}"
    out_dir = os.path.join(base_dir, save_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "combine.csv")
    df.to_csv(out_path, index=False)

    elapsed = time.time() - start
    return data_count, elapsed, out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process peptide datasets to generate fingerprint features."
    )
    parser.add_argument(
        "--fp_type",
        type=str,
        default="ecfp",
        help="Fingerprint type (e.g., ecfp, maccs)",
    )
    parser.add_argument(
        "--nbits", type=int, default=2048, help="Number of bits for fingerprint"
    )
    parser.add_argument(
        "--radius", type=int, default=3, help="Radius for fingerprint generation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    conv = FP_Converter(type=args.fp_type, nbits=args.nbits, radius=args.radius)

    print("Dataset    | Records | Time (s) | Output Path")
    print("-----------|---------|----------|-------------------------")
    for task, dataset_metadata in DATASET_MAP.items():
        nature = dataset_metadata.get("nature", "unknown")
        if nature == "natural":
            base_dir = dataset_metadata["path"]
            data_count, elapsed, out_path = process_split_files(
                base_dir=base_dir,
                converter=conv,
                fp_type=args.fp_type,
                nbits=args.nbits,
                radius=args.radius,
            )
            print(f"{task:<10} | {data_count:>7} | {elapsed:>8.2f} | {out_path}")
