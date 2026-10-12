# Learned vs population demand in EV charging-site coverage

## Motivation
The paper "Strategic optimization of electric vehicle charging station placement in Romania" (2025) applies a facility-location variant to choose charging sites from candidate locations. Such models need a demand weight per location, and population is a common but crude proxy. This experiment tests whether a learned demand estimate improves the choice of sites.

## Question
Can a facility-location model that uses a learned (gradient-boosting) demand estimate choose charging sites that cover more true EV demand, and stay more stable, than the same model using population as demand?

## Dataset
California Housing block groups (scikit-learn `fetch_california_housing`), used as a stand-in. This is not the paper's dataset. EV demand is simulated by a hidden nonlinear function of income, commuting proxies and population plus noise. Only a random 10% of block groups have noisy "observed usage"; a fixed 25% of block groups are held out and their true demand is used only for scoring.

## Baseline vs improved
- Baseline: greedy maximal-coverage facility location (k = 20 and 50, fixed radius, fixed candidates) with population as demand weight.
- Improved: identical solver and candidates, but demand is a LightGBM prediction fitted on the observed-usage block groups with cross-validated tuning.

Main metric: share of true held-out demand covered, averaged over 10 seeds and both k. Secondary: Jaccard overlap of chosen sites across seeds (plus demand-model rank correlation and R²).

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On simulated EV demand over California Housing block groups, a standard population-weighted maximal-coverage facility-location model covered 61.9% of true held-out demand versus 63.8% for the LightGBM-weighted version, a small gain that came with lower site-set stability (Jaccard 0.59 vs 1.00).

| Metric | Value |
|---|---|
| baseline_score | 0.6189 |
| improved_score | 0.6383 |
| baseline_score_std | 0 |
| improved_score_std | 0.003155 |
| baseline_jaccard_stability | 1 |
| improved_jaccard_stability | 0.5863 |
| baseline_demand_spearman | 0.4151 |
| improved_demand_spearman | 0.9766 |
| improved_demand_r2 | 0.9639 |
| baseline_score_k20 | 0.4701 |
| improved_score_k20 | 0.5101 |
| baseline_jaccard_k20 | 1 |
| improved_jaccard_k20 | 0.5256 |
| baseline_score_k50 | 0.7677 |
| improved_score_k50 | 0.7665 |
| baseline_jaccard_k50 | 1 |
| improved_jaccard_k50 | 0.647 |
| improvement | 0.0194 |

Run on 2026-10-12. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Demand is simulated, so conclusions concern this synthetic setting only. Scoring uses noise-free simulated demand on test block groups; the optimiser sees features of all block groups (never test labels). The 10% observed-usage rows are drawn per seed from the training part of one fixed 75/25 split; seeds 42-51 (10) are used rather than 3. Baseline uses no labels and fixed candidates, so it is identical across seeds (std 0, Jaccard 1 trivially); stability only varies for the learned model. Coverage radius is in approximate degrees (0.1 ~ 10 km). Greedy k=20 sites are the prefix of the k=50 solution.

## Reference

This project is motivated by the following paper:

> A Popa, TI Sîrbu (2025). *Strategic optimization of electric vehicle charging station placement in Romania*. Operations Research Forum. https://link.springer.com/article/10.1007/s43069-025-00461-w

Paper author: Prof. Alexandru Popa, University of Bucharest (alexandru.popa@fmi.unibuc.ro)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
