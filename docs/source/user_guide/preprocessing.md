# Preprocessing

PepBenchmark no longer exposes the legacy `pepbenchmark.preprocess.DatasetPreprocessor`
API. The current workflow is built from focused utilities that let you compute
sequence properties, convert between representations, and assemble datasets with
the dataset-manager modules.

## What To Use Instead

- `pepbenchmark.analyze.fasta_level.compute_peptide_properties` for physicochemical descriptors.
- `pepbenchmark.pep_utils.convert` for FASTA, SMILES, HELM, BiLN, fingerprints, and embeddings.
- `pepbenchmark.dataset_manager.single_dataset` and `pepbenchmark.dataset_manager.ppi_dataset`
  for loading benchmark datasets and attaching prepared features.

## Example

```python
import pandas as pd

from pepbenchmark.analyze.fasta_level import compute_peptide_properties
from pepbenchmark.pep_utils.convert import Fasta2Smiles, Smiles2FP

sequences = ["ACDE", "FGHI"]

properties = compute_peptide_properties(sequences)
smiles = Fasta2Smiles()(sequences)
fingerprints = Smiles2FP(fp_type="Morgan", radius=2, nBits=2048)(smiles)

frame = pd.DataFrame(
    {
        "sequence": sequences,
        "smiles": smiles,
        "fingerprint_dim": [len(fp) for fp in fingerprints],
    }
)
```
