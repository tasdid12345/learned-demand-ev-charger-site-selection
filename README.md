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

## Reference

This project is motivated by the following paper:

> A Popa, TI Sîrbu (2025). *Strategic optimization of electric vehicle charging station placement in Romania*. Operations Research Forum. https://link.springer.com/article/10.1007/s43069-025-00461-w

Paper author: Prof. Alexandru Popa, University of Bucharest (alexandru.popa@fmi.unibuc.ro)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
