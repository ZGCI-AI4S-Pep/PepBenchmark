# Negative Sampling

The negative-sampling package helps you build candidate pools, sample negatives, and validate whether the resulting distribution is compatible with the positives.

## Core Objects

- `SamplingPoolManager`
- `NegSampler`
- `DistributionValidator`

```python
from pepbenchmark.neg_sampler import SamplingPoolManager, NegSampler

pool = SamplingPoolManager(include_sequences=["AAAAAA", "CCCCCC", "DDDDDD"])
sampler = NegSampler(pool.get_sampling_pool(), ["ACDEFG"])
negatives = sampler.sample_negatives(method="random", ratio=1.0, seed=42)
```
