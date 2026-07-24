# Paper Plan — Severity-Aware Conformal Risk Control for Medical LLM Hallucinations

**Target venue:** ML4H (Machine Learning for Health) — **Proceedings track** (archival, PMLR).
**Format:** PMLR/JMLR LaTeX template (NOT ACL). natbib-based, so `\citep`/`\citet` and `references.bib` carry over unchanged; only the document scaffold changes.
**One-line thesis:** A single conformal budget controls hallucinations *on average* and therefore lets dangerous claims slip through; giving each clinical severity tier its own conformal budget restores the dangerous-claim guarantee at a modest, benign-only retention cost.

> **Verify before submission:** exact ML4H Proceedings page limit and template version from the current year's CFP. Recent years: full archival paper in PMLR style. Do not hard-code a page count until confirmed.

---

## Status legend
- ✅ drafted — ⏳ in progress — ⬜ not started — 🔁 needs revision after venue/format switch

## Snapshot
| Section | Status | Source file | Notes |
|---|---|---|---|
| Abstract | ⬜ | — | write last, from Results |
| 1. Introduction | ✅ | `paper.md` | drafted; convert md → tex |
| 2. Related Work | ✅ | `related_work.tex` | done, Overleaf-ready |
| 3. Preliminaries | ✅ | `preliminaries.tex` | setup/notation + CRC (known machinery) |
| 4. Severity-Aware CRC | ✅ | `sac.tex` | Proposition + proof + metformin example |
| 5. Experimental Setup | ✅ | `experimental_setup.tex` | closes `\ref{sec:setup-eval}`; has data table; 2 physician-validation TODOs |
| 6. Results | ⬜ | — | **next** — numbers are ready (see below) |
| 6. Discussion / Limitations | ⬜ | — | |
| 7. Conclusion | ⬜ | — | |
| References | ✅ | `references.bib` | 16 entries; 3 to double-check |

**Format debt:** Intro currently lives in `paper.md` (markdown), Related Work + Method are `.tex`. Since we're going to Overleaf for ML4H, the path forward is **all `.tex`**. Convert the Intro to `introduction.tex` and retire `paper.md` as the source of truth (keep it as a readable mirror if useful).

---

## Paper structure (section by section)

### Abstract (~150–200 words) ⬜
Problem (LLMs hallucinate dangerously in medicine; guarantees are harm-blind) → gap (single budget controls the average, dangerous rate escapes) → method (per-tier conformal budgets) → result (dangerous risk 0.068–0.094 → ≤0.049; benign-only retention cost) → significance (first per-severity-tier claim-level conformal budget for medical QA). Write **last**.

### 1. Introduction ✅ (convert to `.tex`)
Already drafted (6 paragraphs) in `paper.md`: hook (metformin 5000 mg vs. benign fact) → CP/CRC in plain terms → claim-level decomposition + Cherian positioning → the single-budget failure (0.068–0.094 above target) → severity-aware fix as *reallocation* not *tightening* → results preview (≤0.049) + 3 contribution bullets. No em/en dashes (author preference).

### 2. Related Work ✅ (`related_work.tex`)
Three paragraphs + positioning: (i) conformal prediction / risk control for LLMs; (ii) claim-level & medical factuality (FactScore, SAFE, MedScore, Cherian); (iii) harm-aware factuality & safety — **CARE (Bedi 2026)** is the load-bearing anchor: it has the machinery but defers per-severity budgets to future work; we do exactly that. Closes with the "first to assign each clinical severity tier its own conformal budget" claim.

### 3. Method ✅ (`method.tex`)
- 3.1 Setup & notation: score `s(c)`, label `y(c)`, tier `t(c)`, keep-if-`s≥λ`, per-claim loss `1[s≥λ]·1[y=1]` (bounded, monotone).
- 3.2 CRC preliminaries: empirical risk, threshold `inf{λ: (n/(n+1))R̂+B/(n+1)≤α}`, marginal guarantee. Matches `crc.py` exactly (marginal, ÷n).
- 3.3 Severity-aware CRC: per-tier partition + independent calibration + deploy-by-own-tier-threshold; α_dangerous=0.05, α_benign=0.15, α_marginal=0.10; framed as Mondrian/group-conditional.
- **Proposition 1** (per-tier risk control) + 3-line proof (corollary of CRC theorem per exchangeable tier subsequence).
- Remarks: expected-vs-per-split (explains violation rate near ½); exchangeability/clustering (motivates the cluster bootstrap).
- 3.4 Worked example: metformin table (global λ keeps the overdose; SAC flags it AND surfaces more benign truth).

### 4. Experimental Setup ⬜ (NEXT)
Must define, and give `\label{sec:setup-eval}` to the eval subsection (Method already references it):
- **Datasets:** K-QA, LiveQA, MedicationQA (+ note on HealthSearchQA if used). Claim counts per dataset per model.
- **Models graded:** Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3 (the answer-generating models under test).
- **Scoring:** confidence score `s(c)` = model's P(true) per claim; how it's elicited.
- **Labeling protocol:** GPT-4o judge for factuality label `y ∈ {0,1}` + unverifiable (−1, excluded); severity-tier grader (dangerous/benign) with physician-validated rubric; report labeler agreement.
- **Data scope / honesty footnote:** Llama headline @ 425 MedicationQA questions (API quota cutoff; 396 pooled dangerous events, overpowered); Mistral @ full 666 (146 pooled dangerous events). Per-model analysis is independent, so question-count symmetry is not required — state this plainly.
- **Evaluation loop:** `N_SPLITS = 300` calibration/test splits; report mean tier risk, retention, violation rate, AUROC.
- **§ Cluster bootstrap (`sec:setup-eval`):** resample whole *questions* with replacement (claims within a question are correlated), recompute risk, take 2.5/97.5 percentiles → 95% CIs.
- **Pooling:** within-model across datasets (scores comparable within a model, not across models).

### 5. Results ⬜ (numbers ready — see table below)
- **Main result:** global CRC dangerous risk exceeds 0.05 on every model×dataset (0.068–0.094, CIs above target); SAC brings it to ≤0.049 with CIs straddling from below.
- **Retention cost:** benign-side only; dangerous retention preserved (report SAC dangerous/benign retention).
- **Violation rate:** global ≈1.00, SAC ≈0.42–0.49 — consistent with the *expectation*-level guarantee (point back to Method remark).
- **Score ablation:** validity is method-borne — even a random score holds the dangerous tier within budget; only retention suffers.
- Figures: (a) per-tier risk global vs. SAC with CIs; (b) risk–retention trade-off; (c) violation-rate bars.

### 6. Discussion / Limitations ⬜
- Guarantee is on expected risk, not per-split (be upfront).
- Claim-level exchangeability idealized (nesting in questions) → cluster bootstrap.
- Severity tiers depend on the grading rubric / judge model; two tiers is a coarse taxonomy.
- Single judge (GPT-4o) for factuality; MedicationQA Llama cap.
- Generalization beyond three QA datasets / two 7–8B models.

### 7. Conclusion ⬜
Restate the reframing (harm-blind → harm-stratified guarantee), the empirical restoration of the dangerous target, and the invitation to richer severity taxonomies / clinician-in-the-loop budgets.

---

## Key results (locked numbers)

α: dangerous **0.05**, benign **0.15**, marginal **0.10**. `N_SPLITS=300`. CIs = question-level cluster bootstrap.

| Model (pooled) | AUROC | Global dRisk [95% CI] | SAC dRisk [95% CI] | SAC dRet | Violation g / s | Dangerous events |
|---|---|---|---|---|---|---|
| **Llama-3-8B** (headline) | 0.851 | 0.068 [0.051, 0.093] | 0.048 [0.044, 0.052] | 0.798 | 1.00 / 0.42 | 396 |
| **Mistral-7B** | 0.846 | 0.094 [0.078, 0.104] | 0.049 [0.044, 0.053] | 0.944 | 1.00 / 0.49 | 146 |

Story holds on both: global CI entirely above 0.05; SAC straddles 0.05 from below. Optional stricter internal α=0.04 variant (SAC CIs tip slightly over 0.05: Llama 0.052, Mistral 0.053) — mention only if a reviewer pushes.

---

## Assets

- **Code:** `sac/` package — `crc.py` (CRC + stratified + metrics), `ablation.py` (`run_ablation` → all reported metrics), `cache.py`, `validate.py`.
- **Notebook:** `notebooks/severity_aware_pipeline.ipynb` — full pipeline, Colab + local (Part 4 = pooled tables + cluster-bootstrap CIs).
- **Graded data:** `results/graded_claims__*.jsonl` (Llama MedicationQA 425q/10,201 claims; Mistral full 666q/5,357 claims; K-QA + LiveQA for both).

---

## Open tasks (in order)
1. ⬜ **Experimental Setup** (`experimental_setup.tex`) — closes `\ref{sec:setup-eval}`.
2. ⬜ **Results** (`results.tex`) + generate the 3 figures from the notebook.
3. ⬜ **Discussion/Limitations**, **Conclusion**.
4. ⬜ **Abstract** (last).
5. 🔁 Convert Introduction `paper.md` → `introduction.tex`; assemble `main.tex` in the ML4H/PMLR template.
6. ⬜ Bib verification pass: `abbasiyadkori2024conformal` author order, `huang2025medscore` first author, `xu2024conformal` exact title.
7. ⬜ Confirm ML4H Proceedings page limit + template version from the current CFP.
8. ⬜ (Optional) add α=0.04 stricter-variant cell to the notebook for the appendix.

## Risks / watch-items
- **Format switch cost:** ACL → PMLR is low-risk (natbib), but rebuild `main.tex` fresh from the ML4H template rather than porting an ACL preamble.
- **Llama data cap** is the most likely reviewer question — pre-empt it honestly in Setup (independent per-model analysis; 396 events is overpowered).
- **Two-tier taxonomy** is a deliberate scope choice; frame coarseness as future work, not a gap.
