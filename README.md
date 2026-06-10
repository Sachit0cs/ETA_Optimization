# Optimizing Delivery ETAs with Graph-Based Network Intelligence

A graph-based intelligence system for Delhivery's hub-and-spoke logistics network.
The project models ~1,600 facilities and ~2,500 corridors as a directed weighted
graph, uses that structure to beat a trip-feature ETA baseline, audits which hubs
and corridors systematically cause delays, prices the FTL-vs-Carting route-type
decision per dispatch, and ships everything as an interactive operations console.

**Dataset:** `delivery_data.csv` (144,867 segment-level scan rows / 14,817 trips /
pre-defined `training` / `test` split).

---

## Results at a glance

All eight ETA models are benchmarked in **one file** -- `outputs/model_comparison.csv` --
on the same 4,163 held-out test trips with identical metrics (single source of truth):

| model | features | MAE (min), lower = better | within 15%, higher = better | vs baseline |
|---|---|---|---|---|
| baseline (trip features) | 5 | 76.14 | 35.6% | -- |
| graph_enhanced | 31 | 59.68 | 45.6% | -21.6% MAE, +10.0 pts |
| graphsage_xgb | 21 | 63.04 | 44.1% | -17.2% MAE, +8.5 pts |
| graphsage_lgbm | 21 | 63.23 | 48.1% | -17.0% MAE, +12.5 pts |
| graphsage_mlp | 21 | 89.60 | 31.4% | (negative -- reported honestly) |
| graphsage_residual | 31 | 61.89 | 40.6% | -18.7% MAE, +5.1 pts |
| node2vec_xgb | 69 | 58.69 | 44.9% | -22.9% MAE, +9.4 pts |
| **node2vec_enhanced** | 79 | **57.42** | **45.8%** | **-24.6% MAE, +10.3 pts** |

The graph advantage is **measured, not claimed**: every number comes from the same
evaluation function on the same split, and every graph feature is computed from
**training trips only** (see *Leakage guarantees* below).

Other headline findings:

- OSRM under-predicts door-to-door time on a **typical trip by ~1.95x**; 97.7% of
  corridors exceed the brief's >20% "chronically delayed" threshold -- so the audit
  ranks by *severity x volume*, not the binary flag.
- Against a **calibrated promise** (OSRM x route-type median factor, +20% tolerance)
  the network still breaches on **29.6% of dispatches**, concentrated in a small set
  of structural hubs (Bhiwandi, Mumbai Hub, Gurgaon Bilaspur, Kolkata Dankuni, Pune).
- The route-type review finds ~**150 Carting-to-FTL upgrades** (~118 min faster each
  for ~Rs 1,300) and ~**2,000 FTL-to-Carting downgrades** that save ~Rs 1,800 each
  with no material time loss (in-support recommendations only).

---

## Repository structure

```
+-- build_graph.py               Task 1 -- graph construction & data pipeline
+-- visualize_graph.py           Task 2 -- network/bottleneck/corridor visualizations
+-- eda.py / eda_utils.py        Auto-generated EDA report (sections self-activate)
+-- graphsage_eta_model.py       GraphSAGE-style node-embedding encoder (PyTorch)
+-- node2vec_eta_model_fixed.py  node2vec embeddings (leakage-fixed; see docstring)
+-- task3_eta_model.py           Task 3 -- unified 8-model ETA benchmark
+-- task4_route_choice.py        Task 4 -- FTL vs Carting decision framework
+-- task5_hub_impact.py          Task 5 (data) -- per-hub SLA breach & revenue at risk
+-- app.py                       Task 5 (deliverable) -- Streamlit operations console
+-- app_lite.py                  Same console, performance build (only the active page renders)
+-- run_app.bat                  One-click launcher for the full console (port 8501)
+-- run_app_lite.bat             One-click launcher for the lite console (port 8502)
+-- .streamlit/config.toml       Dashboard theme
+-- models/                      Trained model artifacts (joblib/pt bundles)
+-- outputs/
    +-- model_comparison.csv     THE benchmark results file (all 8 models)
    +-- node_metrics.csv         Per-hub betweenness/degree/clustering/bottleneck
    +-- edge_weight_table.csv    Per (src, dst, route_type, tod) median delay factor
    +-- corridor_audit.csv       Every corridor with chronic-delay flag
    +-- hub_impact.csv           Per-hub SLA breaches + recoverable + Rs at risk
    +-- route_choice_features.csv         Every test dispatch scored (Task 4)
    +-- route_choice_recommendations.csv  Switch opportunities by distance band
    +-- node_emb_graphsage.csv / node_emb_node2vec.csv   Node embeddings
    +-- eda/EDA_REPORT.md        Auto-generated analysis report + plots
    +-- visualizations/          PNG atlas + 06_interactive_network.html
```

---

## How to run

### 0. Environment

Python >= 3.11. Install dependencies:

```bash
pip install pandas numpy networkx xgboost lightgbm torch gensim scikit-learn matplotlib pyvis joblib streamlit plotly
```

`delivery_data.csv` must sit in the project root.

### 1. Full pipeline (in order)

```bash
python build_graph.py                  # graph + node metrics + corridor audit
python visualize_graph.py              # PNG atlas + interactive HTML map
python task3_eta_model.py --retrain    # all 8 ETA models -> outputs/model_comparison.csv
python task4_route_choice.py --retrain # route-choice bundle + scored dispatches
python task5_hub_impact.py             # hub SLA-breach / revenue table
python eda.py                          # regenerates outputs/eda/EDA_REPORT.md
```

Without `--retrain`, `task3_eta_model.py` and `task4_route_choice.py` reuse cached
models from `models/` and just re-report -- useful for fast verification.
`node2vec_eta_model_fixed.py` can also be run standalone to regenerate only the
node2vec embeddings.

### 2. The operations console (two builds)

```bash
python -m streamlit run app_lite.py    # LITE build -- recommended day to day (http://localhost:8501, or 8502 via the .bat)
python -m streamlit run app.py         # full build (http://localhost:8501)
```

(or double-click `run_app_lite.bat` / `run_app.bat` on Windows -- the .bat files
pin the lite build to port 8502 so both can run side by side). Use
`python -m streamlit` rather than the bare `streamlit` command -- pip
user-installs often don't put the `streamlit` executable on PATH.

Both builds show the same five views computed from the same artifacts with the
same formulas (verified identical to the digit). The difference is architecture --
the lite build exists because the full one re-renders *everything* on *every*
interaction:

| | `app.py` (full) | `app_lite.py` (lite) |
|---|---|---|
| Page rendering | `st.tabs` -- all 5 tabs re-render on every interaction | `st.navigation` -- only the active page runs |
| Slider moves | full-script rerun, decision rule recomputed twice | `@st.fragment` -- only that section reruns; results cached per parameter combo |
| Startup imports | xgboost / sklearn / matplotlib load up front | lazy -- models load on first Route Advisor / Memo visit |
| Interactive map | re-embedded on every rerun, physics never stops | opt-in; physics capped at 350 iterations, switched off once the layout settles, straight edges (halves the simulation bodies) |

Five views:

| Tab | What it does |
|---|---|
| **Command Center** | Network KPIs, SLA-breach pareto by hub, corridor delay distribution, betweenness-vs-delay chokepoint map |
| **Network Map** | The interactive vis.js graph (pan/zoom/hover) + the full static atlas |
| **Model Lab** | All 8 models, user-set priority slider (MAE vs within-15%) and an inference-speed gate -- the recommended model updates live |
| **Route Advisor** | The Task-4 framework with **tunable economics** (value-of-time Rs/min, tail-risk weight, bottleneck uplift, freight rates); per-dispatch trade-off scatter, distance-band shares, a single-dispatch simulator, and buttons to re-run the Task-4 pipeline (with optional retrain) |
| **Strategy Memo** | The Task-5 memo that **rewrites itself** as you change the cost-per-breach, upgrade benchmark and ranking; hub interventions, chronic-corridor table, one-click download |

Neither console reads the 55 MB raw CSV -- both run entirely off the precomputed
artifacts (model predictions are cached; the sliders only re-price the decision
rule). If the console feels slow, use the lite build: it renders one page at a
time, reruns only the section a slider belongs to, and defers the model bundle
and the network map until they are actually asked for.

---

## Methodology

### Task 1 -- Graph construction (`build_graph.py`)

- **Leg extraction.** `is_cutoff == False` rows are the departure scan of each OD
  leg; at that point `actual_time` / `osrm_time` cover the full remaining leg, so
  `factor = actual/osrm` is the per-leg delay ratio.
- **Edges.** One directed edge per corridor. The scalar weight is the trip-count-
  weighted median delay factor; a per-edge lookup table stratifies it by
  **route_type x time-of-day bucket** (night/morning/afternoon/evening) as the brief
  requires. Medians (not means) because the factor distribution is heavy-tailed.
- **Sparse-corridor fallback.** 42% of corridors have <5 trips; their medians are
  one-outlier-fragile, so they fall back to the route-type median (flagged
  `is_sparse`) instead of polluting the bottleneck ranking.
- **Leakage guard.** *Everything* (edge weights, node metrics, embeddings) is
  computed from `data == "training"` rows only, then merged onto both splits.

### Task 2 -- Bottleneck & corridor audit (`build_graph.py`, `visualize_graph.py`)

- **Betweenness centrality is computed unweighted** -- networkx treats edge weight as
  a *distance*, so weighting by delay factor would make slow corridors look "far",
  shortest paths would avoid them, and the metric would invert the meaning of a
  chokepoint. Delay severity is captured separately.
- **Bottleneck score** = betweenness x average incoming delay factor: structurally
  critical *and* chronically slow.
- Corridors with median factor > 1.2 are flagged chronically delayed and ranked by
  *excess delay x trip volume* (the binary flag alone is uninformative at 97.7%).

### Task 3 -- Graph-enhanced ETA models (`task3_eta_model.py`)

- **Trip-level evaluation.** Raw rows are segment-level and cumulative within legs;
  scoring on them double-counts long trips. Rows are collapsed to one per trip
  (targets/features rebuilt from per-hop `segment_*` sums), matching the brief's
  per-trip "% within 15%" metric.
- **Log target.** `actual_time` has skew 3.37, so direct-ETA tree models train on
  `log1p(actual_time)` and invert (with a log-space clip) at prediction time.
- **Feature sets.** baseline = `osrm_time, osrm_distance, is_ftl, hour, dow`;
  plus per-hub graph metrics (betweenness, degrees, avg incoming delay, bottleneck
  score) for src/dst; plus learned embeddings:
  - **GraphSAGE** (`graphsage_eta_model.py`): a small mean-aggregating encoder over
    structural node features, trained to predict hub congestion. Target-derived
    inputs are deliberately excluded from the encoder.
  - **node2vec** (`node2vec_eta_model_fixed.py`): biased second-order random walks
    (p=q=1) on the training-only corridor graph + skip-gram. Implemented in-process
    (networkx + numpy + gensim) because the `node2vec` PyPI package pins numpy<2.
- **Eight models, one protocol.** XGB / LightGBM / MLP / residual heads over the
  feature sets above, all scored by the same `evaluate()` (MAE + within-15% with a
  guard on non-positive actuals) on the same test split, written to
  `outputs/model_comparison.csv`. The MLP's negative result is kept in the table.
- **History.** An earlier notebook benchmarked node2vec at raw-row level with graph
  aggregates computed over train+test (leakage). It was deleted;
  `node2vec_eta_model_fixed.py`'s docstring preserves the full bug audit.

### Task 4 -- FTL vs Carting decision framework (`task4_route_choice.py`)

- **Why counterfactual models:** only 23 of 2,481 corridors were ever run on *both*
  route types, so route choice cannot be answered by lookup. A **T-learner** trains
  separate XGBoost time models on FTL-only and Carting-only legs and predicts every
  dispatch under both.
- **Tail risk:** two additional **p90 quantile models** (`reg:quantileerror`,
  alpha=0.9) price reliability, not just expected minutes.
- **Decision rule (per dispatch):**
  `benefit_FTL = VoT_eff x (delta_time + risk_weight x delta_p90)` vs `delta_cost`,
  where `VoT_eff` rises with the source hub's bottleneck score (reliability is worth
  more out of a fragile hub) and costs come from an explicit parametric model
  (`FTL = Rs 2,500 + Rs 30/km`, `Carting = Rs 45/km` -- **assumptions, tunable in
  the app**). The per-dispatch **break-even value-of-time** (Rs/min at which FTL
  becomes worth it) makes the framework robust to whatever real rates are plugged in.
- **Honesty about extrapolation.** Carting is short-haul (median 31 km), FTL is
  long-haul (median 73 km, p95 873 km). Recommendations outside the recommended
  route type's observed p5-p95 distance support are flagged `extrapolated` and
  excluded from all headline savings. The 23 co-observed corridors give a weak
  sanity check (48% directional agreement on tiny per-corridor samples) -- reported,
  not hidden.

### Task 5 -- Hub impact & strategy memo (`task5_hub_impact.py`, `app.py`)

- **Calibrated SLA definition.** 95% of legs exceed raw OSRM by >20%, so raw-OSRM
  breaches cannot differentiate hubs. The promise an ops team would actually quote
  is `OSRM x route-type median factor`; a breach is a dispatch >20% over that.
  Network breach rate: 29.6%, with hub-level rates ranging 14-57%.
- **Attribution & upgrade model.** Breaches are attributed to the dispatching
  (source) hub. An upgraded hub is assumed to still breach at the benchmark rate
  (network median or best-quartile -- selectable); only the **excess** is counted as
  recoverable. Revenue at risk = breaches x cost-per-breach (default Rs 500,
  explicitly an assumption, adjustable in the memo tab).
- The memo itself is generated live in the dashboard with the chosen assumptions
  and downloadable as Markdown.

---

## Leakage & reproducibility guarantees

1. Graph aggregates (edge weights, node metrics, GraphSAGE + node2vec embeddings)
   are computed **from training rows only** and merged onto both splits.
2. Target-derived node features are excluded from embedding-encoder inputs.
3. One shared evaluation function for every model; non-positive actuals masked
   identically everywhere.
4. Seeds fixed everywhere (numpy / torch / xgboost / gensim with single-worker
   skip-gram), so `--retrain` reproduces the committed numbers.
5. The cached fast path (`python task3_eta_model.py` without `--retrain`) re-emits
   a byte-identical `model_comparison.csv` -- verified.

## Key assumptions (all surfaced as dashboard controls)

| Assumption | Default | Where to change |
|---|---|---|
| FTL cost | Rs 2,500 + Rs 30/km | Route Advisor tab / `task4_route_choice.DEFAULT_PARAMS` |
| Carting cost | Rs 45/km | same |
| Value of time | Rs 20 per delivery-minute | same |
| Tail-risk weight | 0.5 x p90 minutes | same |
| Cost per SLA breach | Rs 500 | Strategy Memo tab / `task5_hub_impact.REVENUE_PER_BREACH` |
| SLA tolerance | +20% over calibrated promise | `DELAY_THRESHOLD` |

## Known limitations

- No real cost or revenue columns exist in the data -- the rupee figures are
  decision-support estimates under documented assumptions, not accounting numbers.
- Counterfactual route-type predictions outside a route type's observed distance
  band are extrapolations; they are flagged and excluded from headline claims.
- The GraphSAGE encoder is a deliberately small single-layer approximation
  (mean-aggregation), not a full minibatch GraphSAGE; node2vec embeddings carry
  most of the learned-graph lift here.
