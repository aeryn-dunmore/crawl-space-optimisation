# crawl-space-optimisation

Code and data accompanying the paper:

> A. Dunmore, K. Gupta, E. Wen, A. Nassani, R. Wang, and M. Billinghurst, "Choices, Choices, Choices: Features, Datasets, and Optimisation for Speech Emotion Recognition," *IEEE Transactions on Affective Computing*, 2026, doi: [10.1109/TAFFC.2026.3733654](https://doi.org/10.1109/TAFFC.2026.3733654).

If you use any of these files, please cite the paper above.

## Contents

| File | Description |
| --- | --- |
| `cso.py` | Reference implementation of Crawl Space Optimisation (CSO), as described in Algorithm 1 of the paper. |
| `example_run.py` | Minimal working example: CSO tuning a Random Forest regressor, plus a sweep over experimental settings with every evaluation logged to CSV. Runs on synthetic data with no downloads. |
| `subject debate timestamps.csv` | Corrected speaking-turn timestamps for the K-EmoCon debates (see below). |
| `Datasets/pycochleagram/` | A copy of the pycochleagram module, used to generate cochleagrams. |

## Crawl Space Optimisation

CSO is a lightweight, single-solution optimiser in the iterated-local-search family:

1. Draw a random start point from each hyper-parameter's initial range.
2. Perturb every hyper-parameter proportionally, `x ← x + x·u` with `u ~ U(−r, +r)`. Integer hyper-parameters are rounded, and all values are clipped to their global bounds.
3. If the new point improves on the current centre, it becomes the new centre.
4. After `m` consecutive attempts without improvement, restart from a new random point.
5. Return the best point found across all restarts.

Paper settings: `r = 0.1` (10%), `m = 5`. The feature-comparison experiments used 100 iterations per combination, and the optimiser comparison (Table III) used 50 iterations per run.

```python
from cso import HyperParameter, crawl_space_optimise

space = [
    HyperParameter("max_depth", start=(3, 10), bounds=(1, 15), integer=True),
    HyperParameter("n_estimators", start=(100, 800), bounds=(50, 1000), integer=True),
]

def objective(point):
    # train a model with point["max_depth"], point["n_estimators"]
    return {"mae": ..., "r2": ...}

result = crawl_space_optimise(objective, space, metric="mae", minimise=True,
                              iterations=50, max_attempts=5, rate=0.1, seed=0)
print(result.best_point, result.best_score)
```

Requirements: Python 3.9+, plus `numpy` and `scikit-learn` for the example.

## K-EmoCon timestamps

The original K-EmoCon labels cover each whole debate, so each participant's labels include periods when they were listening rather than speaking. `subject debate timestamps.csv` gives the start and end of each speaking turn per participant: as minutes.seconds, as a readable duration, in seconds, and as sample indices (frames = seconds × 22,050, the sample rate of the processed K-EmoCon audio).

The K-EmoCon dataset is available from [figshare](https://springernature.figshare.com/articles/dataset/Metadata_record_for_K-EmoCon_a_multimodal_sensor_dataset_for_continuous_emotion_recognition_in_naturalistic_conversations/12618797). If you use these timestamps, please cite both the paper above and:

> C. Y. Park *et al.*, "K-EmoCon, a multimodal sensor dataset for continuous emotion recognition in naturalistic conversations," *Scientific Data*, vol. 7, no. 1, p. 293, 2020. https://www.nature.com/articles/s41597-020-00630-y

## pycochleagram

The original module is available at https://github.com/mcdermottLab/pycochleagram. Please cite it if you use it.
