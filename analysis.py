"""EV charging-site selection: population-weighted vs LightGBM-demand-weighted max-coverage (greedy)."""
import json, os, sys, time, itertools
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, SEEDS, KS = 42, list(range(42, 52)), (20, 50)
RADIUS, N_CAND, OBS_FRAC, TEST_SIZE = 0.10, 3000, 0.10, 0.25  # radius in ~degrees

# ---- data (retry with backoff) ----
df = None
for attempt in range(5):
    try:
        df = fetch_california_housing(as_frame=True, data_home="data").frame
        break
    except Exception as e:
        print("download failed:", e); time.sleep(2 ** attempt)
if df is None:
    sys.exit("could not download data")
feats = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
X = df[feats].reset_index(drop=True)
N = len(X)
print("rows:", N)

# ---- hidden true EV demand (nonlinear in income, commute proxies, population) ----
adopt = 1 / (1 + np.exp(-1.2 * (X.MedInc - 4)))
commute = np.exp(-(X.AveOccup.clip(1, 6) - 3) ** 2 / 2) + 0.5 * (X.HouseAge < 20)
true_d = ((X.Population ** 0.7) * adopt * (0.5 + commute)).to_numpy()
true_d = true_d / true_d.mean()

# ---- single train/test split; test true demand used only for scoring ----
train_idx, test_idx = train_test_split(np.arange(N), test_size=TEST_SIZE, random_state=SEED)
test_mask = np.zeros(N, bool); test_mask[test_idx] = True

# ---- shared coverage structure: fixed candidate set, fixed radius ----
xy = np.c_[X.Longitude * 0.8, X.Latitude]  # rough km-correction for longitude
cand = np.sort(np.random.default_rng(SEED).choice(N, N_CAND, replace=False))
lists = cKDTree(xy).query_ball_point(xy[cand], RADIUS)
indptr = np.r_[0, np.cumsum([len(l) for l in lists])]
M = sp.csr_matrix((np.ones(indptr[-1]), np.concatenate(lists), indptr), shape=(N_CAND, N))
print("mean points covered per candidate:", M.getnnz() / N_CAND)

def greedy(w, k):
    """Standard greedy maximal coverage: pick site with largest uncovered weight."""
    unc, chosen = w.astype(float).copy(), []
    for _ in range(k):
        gain = M @ unc
        gain[chosen] = -1
        j = int(np.argmax(gain)); chosen.append(j)
        unc[M[j].indices] = 0
    return chosen

def score(chosen):
    """Share of true held-out demand within radius of chosen sites."""
    cov = np.asarray(M[chosen].sum(axis=0)).ravel() > 0
    return true_d[test_mask & cov].sum() / true_d[test_mask].sum()

def jaccard(sets):
    return float(np.mean([len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]))

grid = {"n_estimators": [100, 300], "learning_rate": [0.03, 0.1], "num_leaves": [8, 16, 31],
        "min_child_samples": [10, 30]}
res = {m: {k: [] for k in KS} for m in ("base", "imp")}
sites = {m: {k: [] for k in KS} for m in ("base", "imp")}
rho = {"base": [], "imp": []}; r2_imp = []

# baseline weights: population (data-only, no labels)
w_base = X.Population.to_numpy()
ch_base = greedy(w_base, max(KS))
for s in SEEDS:
    rng = np.random.default_rng(s)
    obs = rng.choice(train_idx, int(OBS_FRAC * N), replace=False)  # block groups with observed usage
    y_obs = np.clip(true_d[obs] + rng.normal(0, 0.3 * true_d.std(), len(obs)), 0, None)
    search = RandomizedSearchCV(LGBMRegressor(random_state=s, verbose=-1, n_jobs=1), grid, n_iter=6,
                                cv=3, scoring="neg_mean_squared_error", random_state=s, n_jobs=1)
    search.fit(X.iloc[obs], y_obs)  # tuning by CV on observed-usage rows only
    pred = np.clip(search.predict(X), 0, None)
    ch_imp = greedy(pred, max(KS))
    r2_imp.append(r2_score(true_d[test_idx], pred[test_idx]))
    rho["imp"].append(spearmanr(pred[test_idx], true_d[test_idx])[0])
    rho["base"].append(spearmanr(w_base[test_idx], true_d[test_idx])[0])
    for k in KS:
        for m, ch in (("base", ch_base), ("imp", ch_imp)):
            res[m][k].append(score(ch[:k])); sites[m][k].append(set(ch[:k]))
    print(f"seed {s}: base={np.mean([res['base'][k][-1] for k in KS]):.4f} "
          f"imp={np.mean([res['imp'][k][-1] for k in KS]):.4f} best={search.best_params_}")

per_seed = {m: np.mean([res[m][k] for k in KS], axis=0) for m in res}  # mean over k, per seed
metrics = {
    "baseline_score": float(per_seed["base"].mean()), "improved_score": float(per_seed["imp"].mean()),
    "baseline_score_std": float(per_seed["base"].std()), "improved_score_std": float(per_seed["imp"].std()),
    "baseline_jaccard_stability": float(np.mean([jaccard(sites["base"][k]) for k in KS])),
    "improved_jaccard_stability": float(np.mean([jaccard(sites["imp"][k]) for k in KS])),
    "baseline_demand_spearman": float(np.mean(rho["base"])), "improved_demand_spearman": float(np.mean(rho["imp"])),
    "improved_demand_r2": float(np.mean(r2_imp)),
}
for k in KS:
    metrics[f"baseline_score_k{k}"] = float(np.mean(res["base"][k]))
    metrics[f"improved_score_k{k}"] = float(np.mean(res["imp"][k]))
    metrics[f"baseline_jaccard_k{k}"] = jaccard(sites["base"][k])
    metrics[f"improved_jaccard_k{k}"] = jaccard(sites["imp"][k])
metrics["improvement"] = metrics["improved_score"] - metrics["baseline_score"]
if not all(np.isfinite(v) for v in metrics.values()) or metrics["baseline_score"] <= 0:
    sys.exit("no meaningful metrics")
print(json.dumps(metrics, indent=2))

os.makedirs("figures", exist_ok=True)
plt.bar(["baseline (population)", "improved (LightGBM)"], [metrics["baseline_score"], metrics["improved_score"]],
        yerr=[metrics["baseline_score_std"], metrics["improved_score_std"]])
plt.ylabel("share of held-out true demand covered"); plt.tight_layout()
plt.savefig("figures/coverage.png", dpi=120)

results = {
    "question": "Can a facility-location model that uses a learned (gradient-boosting) demand estimate choose charging sites that cover more true EV demand, and stay more stable, than the same model using population as demand?",
    "method": "Greedy maximal-coverage (k=20 and 50, fixed radius, fixed 3000 candidate block groups) on California Housing block groups with simulated hidden EV demand; baseline weights demand by population, improved by a LightGBM model fitted (CV-tuned) on a random 10% of block groups with noisy observed usage. Scored on true demand of 25% held-out block groups; 10 seeds.",
    "dataset": {"name": "California Housing (scikit-learn) with simulated EV demand", "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html",
                "n_rows": int(N), "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "share of true held-out demand covered (mean over k=20,50 and 10 seeds)",
    "higher_is_better": True,
    "baseline": "Greedy maximal-coverage facility location with population as demand weight.",
    "improved": "Same greedy solver and candidates, but demand weight is a CV-tuned LightGBM prediction fitted on observed-usage block groups.",
    "metrics": metrics,
    "notes": "Demand is simulated, so conclusions concern this synthetic setting only. Scoring uses noise-free simulated demand on test block groups; the optimiser sees features of all block groups (never test labels). The 10% observed-usage rows are drawn per seed from the training part of one fixed 75/25 split; seeds 42-51 (10) are used rather than 3. Baseline uses no labels and fixed candidates, so it is identical across seeds (std 0, Jaccard 1 trivially); stability only varies for the learned model. Coverage radius is in approximate degrees (0.1 ~ 10 km). Greedy k=20 sites are the prefix of the k=50 solution.",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("done")
