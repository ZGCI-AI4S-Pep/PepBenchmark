# Redundancy Analysis

Use the redundancy package when you want to quantify duplicate structure or remove near-duplicates before downstream evaluation.

```python
import numpy as np
from pepbenchmark.redundancy import RedundancyAnalyzer

report = RedundancyAnalyzer(
    ["ACDE", "ACDE", "AAAA"],
    np.array([[1.0, 1.0, 0.25], [1.0, 1.0, 0.25], [0.25, 0.25, 1.0]]),
).compute_metrics(thresholds=(0.8,))
```
