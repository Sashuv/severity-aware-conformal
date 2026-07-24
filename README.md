# Severity-Aware Conformal Prediction

Risk-controlled selective prediction for medical question answering. A generator answers each
question, the answer is split into atomic claims, and each claim is scored with P(true).
Conformal Risk Control (CRC) then decides which claims to keep so that the error rate stays
within a chosen budget. The contribution is **severity-aware** CRC: dangerous claims are held to
a stricter budget than benign ones, instead of a single global threshold that treats every error
the same.

## Repository layout

| Path | What it holds |
| --- | --- |
| `sac/` | The core Python package: decomposition, P(true) scoring, judge grading, severity tagging, CRC calibration, validation, and ablation. This is where the methods live. |
| `notebooks/severity_aware_pipeline.ipynb` | The end-to-end driver. Runs the whole study top to bottom on one GPU, one `(dataset, model)` pair at a time, then pools every run for the final analysis. |
| `notebooks/` (phase1/2/3, score_ablation) | Per-phase notebooks that call into `sac/`. |
| `results/` | Cached `scored_claims__*.jsonl` and `graded_claims__*.jsonl` for each dataset and model, so the analysis can be rebuilt without re-running generation. |
| `tests/` | Unit tests for the `sac/` package. |
| `paper/`, `figures/`, `acl_latex.tex` | Paper draft (tex + markdown) and figure/table generation. |

## Pipeline

1. **Generate and score** (GPU): the generator answers each question, the answer is decomposed
   into atomic claims, each scored with P(true).
2. **Grade and tag severity** (OpenAI judge): each claim is labeled true / hallucination /
   unverifiable and tagged with a danger tier.
3. **Validate and calibrate** (CPU): Gate 2 AUROC, then global versus severity-aware CRC.
4. **Pooled analysis** (CPU): pool each model across all datasets and report the
   global-versus-severity-aware headline with cluster-bootstrap confidence intervals.

Datasets share the K-QA `Must_have` / `Nice_to_have` statement format: K-QA plus the MedLFQA
subsets `liveqa`, `medicationqa`, and `healthsearchqa`.

## Running it

Launch Jupyter from the repo root (so `sac/` imports) and open
`notebooks/severity_aware_pipeline.ipynb`. Set `DATASET` and `GEN_MODEL` in the config cell, then
Run All. Parts 1 and 2 need a CUDA GPU and an OpenAI API key; Part 4 (analysis) reads only
`results/graded_claims__*.jsonl` and runs anywhere with no GPU or API.

Credentials are resolved at runtime from an environment variable, then a Colab Secret, then an
interactive prompt, and are never written to disk.

```bash
pip install -r requirements.txt          # core package
pip install -r requirements-notebook.txt  # extras for the notebooks
pytest                                     # run the test suite
```
