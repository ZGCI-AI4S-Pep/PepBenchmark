# Feature Conversion

PepBenchmark ships converters for FASTA, SMILES, HELM, BiLN, fingerprints, embeddings, and graphs.

## Common Converters

- `Fasta2Smiles`
- `Fasta2Helm`
- `Fasta2Biln`
- `Smiles2FP`
- `Fasta2Embedding`

```python
from pepbenchmark.pep_utils.convert import Fasta2Smiles, Smiles2FP

smiles = Fasta2Smiles()("ACD")
fp = Smiles2FP(fp_type="Morgan", radius=2, nBits=32)(smiles)
```
