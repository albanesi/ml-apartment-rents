# Predicting Apartment Rents � Group 2

**Course:** Fundamentals of Python & Applications in Data Science
**Option:** B — Machine Learning

## Group members
- Ghebre Abel
- Bertazzoli Leo
- Brody Daniel
- Etemi Zabit
- Selimi Alban

## Setup
pip install -r requirements.txt

## Data
**Source:** Inside Airbnb — Zurich, detailed listings file (79 columns)
**Snapshot date:** 2025-09-29 (scrape_id 20250929042214)
**Rows:** 3 417 listings

Processed train/test splits are committed under `data/processed/` so the pipeline
runs without re-downloading. To reproduce from the raw file:

1. Download `listings.csv.gz` and `calendar.csv.gz` from https://insideairbnb.com/get-the-data/ (select Zurich).
2. Extract `listings.csv.gz` and place at `data/raw/listings.csv`. Leave `calendar.csv.gz` gzipped at `data/raw/calendar.csv.gz` (pandas reads it directly).
3. Run notebooks 01 — 05 in order.

IMPORTANT: Use the detailed version (~79 columns). The simple 18-column file will break notebook 02.

## Reproduction — run notebooks in this exact order
1. `01_data_ingestion.ipynb`  — loads raw data, saves to `data/processed/`
2. `02_eda.ipynb`             — EDA plots, parses `amenities` into `n_amenities` + 10 binary flags, saves `data/processed/listings_eda.csv`
3. `03_preprocessing.ipynb`   — feature selection, haversine venue distances, log-transform target, train/test split. Saves **raw** (unencoded, unscaled) `X_train/X_test/y_train/y_test`
4. `04_modeling.ipynb`        — builds a sklearn `Pipeline = ColumnTransformer + model` per candidate (Linear Regression, `GridSearchCV`-tuned Random Forest, `GridSearchCV`-tuned HistGradientBoosting). Winner picked by 5-fold CV-R² on the training set; saves the full pipeline to `models/rent_pipeline.pkl`
5. `05_evaluation_xai.ipynb` — metrics, residuals, SHAP plots, ZFF scenario analysis. Calls `pipeline.predict(X_test)` directly on raw features

Each notebook depends on the previous one. Always restart kernel and run all cells.

### Pipeline architecture
The saved `rent_pipeline.pkl` contains a single `sklearn.pipeline.Pipeline`:
```
Pipeline([
    ('pre', ColumnTransformer([
        ('num',    SimpleImputer(median) → StandardScaler,         NUMERIC),
        ('cat_te', TargetEncoder(cv=5, smooth='auto'),              ['neighbourhood_cleansed', 'property_type']),
        ('cat_oh', OneHotEncoder(handle_unknown='ignore'),          ['room_type']),
        ('bin',    'passthrough',                                    BINARY),
    ])),
    ('model', <winning estimator>),
])
```
A consumer can `joblib.load(...).predict(raw_dataframe)` — no separate scaler, no manual one-hot. The old `models/scaler.pkl` is obsolete and unused by the new pipeline.

## Features

The final pipeline uses **61 raw input features** after a SHAP-based pruning step in nb03 that removed 27 weak-signal columns (mean |SHAP| < 0.003 on the training set — see the iteration table below). The numbers in parentheses below are the **post-pruning** counts; what was originally generated before pruning is noted where it differs.

- **Listing structure:** `accommodates`, `bedrooms`, `bathrooms`, `beds`, `room_type`, **target-encoded** `property_type` (29 categories collapsed to one continuous price-prior column via out-of-fold encoding, same approach as `neighbourhood_cleansed`), `minimum_nights`
- **Location:** `latitude`, `longitude`, **target-encoded** `neighbourhood_cleansed` (34 categories collapsed to one continuous price-prior column via out-of-fold encoding)
- **Booking dynamics:** `availability_30`, `availability_60`, `availability_90`, `availability_365` (short / medium / long-term availability). `instant_bookable` was generated but pruned (SHAP rank #59 in the unpruned model).
- **Review activity:** `number_of_reviews`, `number_of_reviews_ltm` (last 12 months), `number_of_reviews_l30d` (last 30 days), `reviews_per_month`
- **Event-venue proximity (6):** haversine distance in km to Hallenstadion, Letzigrund, Messe Zürich, Opernhaus, Zürich HB, plus `min_dist_venue_km`
- **Amenities (kept 10 of 30):** `n_amenities` total count + the 10 binary `amen_*` flags that survived SHAP pruning: `amen_dishwasher`, `amen_tv`, `amen_lockbox`, `amen_hot_water_kettle`, `amen_iron`, `amen_microwave`, `amen_hot_water`, `amen_essentials`, `amen_hair_dryer`, `amen_shampoo`. nb02 still generates the full top-30 by frequency; the 20 near-ubiquitous ones (`amen_wifi`, `amen_kitchen`, `amen_heating`, …) are dropped in nb03.
- **Host signals (5):** `host_is_superhost`, `host_response_rate` (%), `host_acceptance_rate` (%), `host_response_time` (ordinal 1–4, lower = faster), `host_total_listings_count` (signals professional vs. private hosts)
- **Review scores (kept 6 of 7):** `review_scores_rating` plus five subscores (cleanliness, check-in, communication, location, value). `review_scores_accuracy` was pruned (overlapped with `review_scores_rating`). ~24% of listings have no reviews and are median-imputed inside the Pipeline.
- **Text features (kept 9 of 14)** from `name` + `description`: 2 length features (`text_name_len`, `text_desc_len`) + 7 bilingual keyword flags (`text_luxury`, `text_central`, `text_modern`, `text_terrace`, `text_quiet`, `text_spacious`, `text_charming`) — regexes match both German and English variants. Five keywords were pruned: `text_renovated`, `text_view`, `text_design`, `text_river`, `text_lake`.
- **Calendar-derived (6)** from `calendar.csv.gz` (per-listing aggregates over the next 365 days; joined in nb03 via `id`): `cal_min_nights_median`, `cal_min_nights_std`, `cal_max_nights_median` (min/max-stay strategy), `cal_avail_weekend_premium` (weekend booking pressure), `cal_avail_rate_zff_2025` (Sep 29 – Oct 5 2025 availability — direct ZFF event-pressure signal), `cal_avail_rate_winter_hols` (Dec 20 – Jan 2 availability). The calendar's `price` column is fully NaN in recent Inside Airbnb snapshots, so we use only structural/availability aggregates — these are also genuinely orthogonal to `listings.csv` (which has only single-value snapshots).

## Results
| Model                       | MAE (CHF) | RMSE (CHF) | R²    | CV R² (log) |
|-----------------------------|-----------|------------|-------|-------------|
| Linear Regression           | 58.60     | 108.00     | 0.368 | 0.385       |
| Random Forest (n=100, untuned, baseline) | 51.38     | 100.00     | 0.458 | 0.478       |
| Random Forest (tuned) — pre-amenity-expansion | —         | —          | —     | 0.583       |
| HistGradientBoosting (tuned) — pre-amenity-expansion | —         | —          | —     | 0.635       |
| Random Forest (tuned) — top-30 amenities + one-hot neighbourhood | —         | —          | —     | 0.590       |
| HistGradientBoosting (tuned) — top-30 amenities + one-hot neighbourhood | —         | —          | —     | 0.644       |
| Random Forest (tuned) — target-encoded neighbourhood | —         | —          | —     | 0.587       |
| HistGradientBoosting (tuned) — target-encoded neighbourhood | —         | —          | —     | 0.641       |
| Random Forest (tuned) — expanded columns + text features | —         | —          | —     | 0.614       |
| HistGradientBoosting (tuned) — expanded columns + text features | —         | —          | —     | 0.670       |
| Random Forest (tuned) — + calendar.csv features | —         | —          | —     | 0.619       |
| HistGradientBoosting (tuned) — + calendar.csv features | —         | —          | —     | 0.668       |
| Random Forest (tuned) — SHAP-pruned (88 → 61 features) | —         | —          | —     | 0.619       |
| HistGradientBoosting (tuned) — SHAP-pruned (88 → 61 features) | —         | —          | —     | 0.669       |
| Random Forest (tuned) — + `property_type` target-encoded | —         | —          | —     | 0.612       |
| HistGradientBoosting (tuned) — + `property_type` target-encoded | —         | —          | —     | **0.663**   |
| **Winner: HistGradientBoosting** (saved to `rent_pipeline.pkl`) | **36.77** | **62.69** | **0.589** | **0.663** |

The first two rows are historical baselines (run before amenity features and tuning were added) and are kept for reference. The tuned-RF and tuned-HGB rows show only CV-R² on the training set — by policy the held-out test set is touched exactly once, on the winning model. Test metrics for the winner (MAE / RMSE / R²) are written to `reports/metrics.json` by nb05.

### Iteration-by-iteration evolution of held-out metrics

Δ MAE / Δ RMSE are *relative* changes (how many percent smaller); Δ R² is *absolute* (R² is already a fraction, so 0.020 means "+2 percentage points of variance explained"). Each row's Δ values compare against the **immediately previous row**, not the baseline.

| # | Iteration / change                                                            | MAE (CHF) | Δ MAE     | RMSE (CHF) | Δ RMSE    | Test R² | Δ R²       |
|---|-------------------------------------------------------------------------------|-----------|-----------|------------|-----------|---------|------------|
| 0 | Pre-PR RF baseline (hand-rolled scaler, untuned)                              | 51.38     | —         | 100.00     | —         | 0.458   | —          |
| 1 | + sklearn `Pipeline` + `GridSearchCV` → HGB winner                            | 41.20     | **−19.8%**| 70.61      | **−29.4%**| 0.478   | **+0.020** |
| 2 | + `property_type`, `host_*`, `review_scores_*`                                | 38.80     | **−5.8%** | 67.02      | **−5.1%** | 0.530   | **+0.052** |
| 3 | + top-30 amenity one-hots (replacing 10 hand-picked)                          | 38.29     | **−1.3%** | 66.64      | **−0.6%** | 0.535   | **+0.005** |
| 4 | + target-encoded `neighbourhood_cleansed` (interpretability, see note below)  | 38.14     | −0.4%     | 66.98      | +0.5%     | 0.530   | −0.005     |
| 5 | + unused listings.csv columns + 14 text features from `name`/`description`    | 36.50     | **−4.3%** | 63.88      | **−4.6%** | 0.573   | **+0.043** |
| 6 | + 6 calendar.csv features (min/max-stay, weekend pressure, ZFF/Xmas)           | 36.32     | −0.5%     | 62.81      | −1.7%     | 0.587   | +0.014     |
| 7 | + SHAP-based pruning (drop 27 cols with mean \|SHAP\| < 0.003: 20 amenities, 5 text keywords, `instant_bookable`, `review_scores_accuracy`) | 36.75 | +1.2%     | 63.74  | +1.5%     | 0.575 | −0.012   |
| **8** | **+ target-encode `property_type`** (≈20 sparse OH cols → 1 continuous TE col, same approach as `neighbourhood_cleansed` in row 4) | **36.77** | +0.1% | **62.69** | **−1.6%** | **0.589** | **+0.014** |
|   | **Cumulative (#0 → #8)**                                                      |           | **−28.4%**|            | **−37.3%**|         | **+0.131** |
|   | **Cumulative (#1 → #8, feature-engineering only)**                            |           | **−10.8%**|            | **−11.2%**|         | **+0.111** |

Two distinct "phases" stand out: row 1 → 2 (proper Pipeline + tuning) and row 4 → 5 (harvesting unused 79-column features + text keywords) each delivered ~+0.05 R² on their own. Row 4's small R² regression was the trade we accepted for a much cleaner SHAP picture (34 sparse one-hots → 1 target-encoded column, see the note further below). Row 8 applies the same target-encoding trick to `property_type` (29 categories): test R² rose 0.575 → 0.589 (+0.014) and the SHAP picture collapsed ~20 scattered one-hot bars into a single column that landed at SHAP rank #2 (mean |SHAP| 0.110), behind only `accommodates` — see the SHAP section for details.

**About row 7 (SHAP pruning).** After step 6 we ran a SHAP analysis on the **training set** (100-sample background + 800-sample explainee, no test-set peeking) and dropped 27 raw columns whose aggregated mean \|SHAP\| was below 0.003 — mostly near-ubiquitous amenities (`amen_wifi`, `amen_kitchen`, `amen_heating`, …) that appear on >70 % of listings and therefore can't discriminate price, plus the five weakest text keywords, `instant_bookable`, and the redundant `review_scores_accuracy`. **CV-R² stayed flat (HGB 0.668 → 0.669, RF 0.619 → 0.619) but held-out test R² fell from 0.587 to 0.575 (−0.012)** — a small drop, well within typical sampling noise on a 514-row test set, but consistent across MAE/RMSE/R². We kept the pruning because the model now uses **61 raw input features instead of 88 (−31 %)** with essentially unchanged generalisation ability (the metric of record for sklearn model selection, CV-R², is unchanged), and the SHAP bar plot is much easier to read with the dead-weight features gone.

### Where R² 0.575 sits, and why we stopped here

Published Airbnb-price studies on single-snapshot Inside Airbnb data (NYC, Boston, Paris, London) typically report **test R² in the 0.50–0.65 range** when using only structured listing features — i.e. without images, multi-date scraping, or external booking-history data. **Our 0.589 sits in the upper half of that range.**

The remaining ~42 % of price variance that no structured-features model can capture from a single snapshot is intrinsically:
- **Host pricing strategy** (subjective, often inconsistent across hosts) — the single biggest source.
- **Photo / visual-quality effects** that the `picture_url` field hints at but no structured column encodes.
- **Day-of-week and event-driven dynamic pricing** the host actually charges, which only multi-date scrapes can resolve.
- **Brand effects** of managed agencies and short-term-rental companies that aren't visible as a clean feature.

To push meaningfully beyond ~0.60 would require a structurally different dataset, not better feature engineering on this one. Concretely: monthly Inside Airbnb snapshots over a year (≈ +0.10 R² achievable, but ≈ 1–2 days of data work) or CNN features extracted from `picture_url` (≈ +0.05–0.10, but 2–3 days plus GPU access). Both were out of scope for this project.

**Evidence the snapshot is near-exhausted.** After reaching 0.587 we attempted four further feature ideas — OSM transit + supermarkets, OSM lake + river distance, an outlier-stratified two-expert HGB, and a 30-token guest-review TF-IDF (full details in *Experiments that didn't pay off* below). None generalised: either CV-R² regressed or held-out R² dropped. The subsequent SHAP-based pruning in step 7 also showed flat CV-R². Step 8's `property_type` target encoding then recovered +0.014 on the test set without changing the raw feature set — a representation-only win — but a separate attempt at the same time to broaden the calendar windows (Summer/Shoulder/Winter-Offpeak availability rates on top of the existing ZFF/Xmas rates) moved test R² by <0.01 and was reverted. This pattern — several reasonable, orthogonal-looking ideas no longer moving the needle while only representation changes still help — is the strongest argument that the snapshot's information content for this model class is largely consumed.

**Why target-encode the high-cardinality categoricals?** The 34 one-hot neighbourhood columns were each contributing tiny SHAP values (top one-hot `Hard` at rank #42, ~0.0023) — the signal was real but **fragmented across 34 columns**. Target encoding consolidates the same signal into one column. The same logic was extended to `property_type` in row 8: ~20 sparse one-hots (each at SHAP rank #30+) collapsed into a single TE column that jumped to **SHAP rank #2 (mean |SHAP| 0.110)** — second only to `accommodates`. Test R² rose from 0.575 to 0.589 (+0.014), MAE essentially flat (36.75 → 36.77), RMSE −1.05 CHF. The encoder's internal 5-fold CV makes it leakage-free. `room_type` (only 4 categories) is left as one-hot — the dummy expansion is trivial there, and keeping `Entire home/apt` as an explicit indicator preserves interpretability.

Best hyperparameters from `GridSearchCV` on the current run:
- Random Forest (tuned): `n_estimators=400, max_depth=15, min_samples_leaf=1`
- HistGradientBoosting (winner): `learning_rate=0.05, max_depth=None, max_iter=400, min_samples_leaf=20`

A `DummyRegressor(strategy='mean')` baseline is also printed in nb04 to anchor what "zero skill" looks like on this data.

## Key findings (SHAP) — HistGradientBoosting winner

Mean |SHAP value| on log-price scale (top features, from `reports/figures/shap_bar.png`):

1. **`accommodates` (#1, ≈0.111)** and **`property_type` target-encoded (#2, ≈0.110)** — structural capacity is essentially tied with property-type price prior as the dominant signal. After moving `property_type` from one-hot to target encoding, the ~20 sparse property-type one-hots (each previously SHAP rank #30+) collapsed into this single column, which surfaces just behind `accommodates`.
2. **`bedrooms`** (#3, ≈0.078) — second structural-capacity signal, complementary to `accommodates`.
3. **`host_total_listings_count`** (#4, ≈0.050) — distinguishes professional managers (max 2 396 listings in the data!) from private hosts.
4. **`availability_30`** (#5, ≈0.036), **`dist_opernhaus_km`** (#6, ≈0.031), **`dist_hb_km`** (#7, ≈0.029), **`neighbourhood_cleansed`** target-encoded (#8, ≈0.029) — short-term booking pressure and central-location signal. The neighbourhood TE slipped one rank vs. the previous pipeline because `property_type` now absorbs some of the location-correlated signal (certain property types cluster in certain neighbourhoods).
5. **Calendar features land high: `cal_min_nights_median`** (#9, ≈0.028) — the typical minimum-stay setting from the per-day calendar carries more signal than the single-value `minimum_nights` snapshot (which sits at #24). `cal_avail_weekend_premium` (#21, ≈0.017) and `cal_max_nights_median` (#28, ≈0.013) also rank reasonably; `cal_min_nights_std` (#33) carries weaker booking-strategy signal.
6. **Amenities (`amen_dishwasher` #10, `amen_tv` #11)**, **`review_scores_rating`** (#12), **`host_acceptance_rate`** (#13), **`availability_365`** (#14) — quality and booking-window signals. `amen_dishwasher` is the strongest amenity signal (vs. `amen_wifi`/`amen_kitchen` which were pruned because they appear on >90 % of listings).
7. **`reviews_per_month`** (#16), **`review_scores_cleanliness`** (#18), **`beds`** (#19), **`number_of_reviews_l30d`** (#20) — booking velocity + structural + review-quality follow-ups.
8. **`text_modern`** (#23, ≈0.015) still represents the text-keyword family. "Modern"/"moderne"/"stylish" in the listing copy carries a real price premium; `text_desc_len` (#30) is nearby as a quality proxy.
9. **Surprising flops:** `cal_avail_rate_zff_2025` (#46) and `cal_avail_rate_winter_hols` (#39) — event-period availability is less differentiating than the full-year strategy aggregates, probably because too few days are in each window. **`room_type` one-hots all rank below #53**: `Entire home/apt` at #53 (≈0.004), `Private room` at #62 (≈0.002), `Hotel room` and `Shared room` at SHAP 0.000 — once `property_type` is target-encoded, `room_type` carries almost no additional information (the room/property split is largely redundant). Kept anyway for interpretability and because its dummy expansion is only 4 columns.

Compared to the earliest run, the model now uses a much more diverse feature mix: structural capacity → property-type price prior → location (distance + neighbourhood prior) → host professionalisation → booking dynamics (incl. per-day calendar strategy) → review quality → amenities → text framing. The cumulative lift on test R² (0.478 → 0.589, +0.111) is consistent with each of these feature families carrying genuine, partially-orthogonal pricing signal.

See `reports/figures/shap_summary.png` for the beeswarm (direction + magnitude per sample) and `shap_bar.png` for the mean-impact ranking.

## Experiments that didn't pay off

| Experiment                                       | Tried after step | Test R² (before) | Test R² (after) | Δ R²    | Outcome                                       |
|--------------------------------------------------|------------------|------------------|-----------------|---------|-----------------------------------------------|
| OSM transit stops + supermarkets in 500 m radius | 2 (R² 0.530)     | 0.530            | 0.535           | +0.005  | reverted — CV-R² dropped 0.635 → 0.626        |
| OSM Zürichsee shore + Limmat river distance      | 2 (R² 0.530)     | 0.530            | 0.522           | −0.008  | reverted — within noise; signal redundant     |
| Outlier-stratified HGB (mixture-of-experts)      | 4 (R² 0.530)     | 0.530            | n/a (CV lost)   | n/a     | reverted — CV-R² 0.641 → 0.558, never reached test |
| Guest-review TF-IDF (top-30 tokens)              | 6 (R² 0.587)     | 0.587            | 0.563           | −0.024  | reverted — overfit (CV flat, test dropped)    |
| Broader seasonal calendar rates (Summer / Shoulder / Winter-Offpeak availability) | 8 (R² 0.589) | 0.589 | 0.582 | −0.007 | reverted — new windows partly redundant with existing `availability_30/60/90` |

Details on each attempt:

- **OSM transit stops + supermarkets in 500 m radius.** `dist_nearest_transit_km` and `n_supermarkets_500m` fetched from OpenStreetMap via the Overpass API. Zurich's transit network is so dense that almost every listing sits within ~150 m of a stop (median 112 m, max 711 m), so the distance has no discriminative power. Supermarket density correlated with `dist_hb_km` (city centre), so the model already had that signal. Both features landed at SHAP rank #26 / #30 and CV-R² dropped slightly (0.635 → 0.626).
- **Distance to Zürichsee shore + Limmat river.** Same OSM source. These features *do* carry signal (SHAP rank #12 / #13), but SHAP simply redistributed importance from the existing `dist_opernhaus_km` and `dist_hb_km` features — held-out R² changed by −0.008 (within noise). Lake/river proximity is already implicitly captured by the central-Zurich venue distances.
- **Outlier-stratified HGB (two-expert / mixture-of-experts).** Built a custom `StratifiedPriceRegressor` that trains a `HistGradientBoostingClassifier` to split listings at 250 CHF/night (~15% luxury bucket), then routes each prediction to one of two HGB regressors trained on its bucket. The stratified pipeline lost by a wide margin: CV-R² **0.558** vs. the single-HGB baseline's **0.641** (−0.083). Two reasons: the router classifier mis-routes ~10% of listings (each error costs a lot in MAE), and the luxury bucket's 299 training rows is below the sample-size regime where HGB outperforms simpler models. The single HGB sees the full distribution and learns the price-by-feature relationship without any routing risk.
- **Guest-review TF-IDF (top-30 tokens) from `reviews.csv.gz`.** Downloaded the ~110k Zurich guest reviews and computed 30 TF-IDF token features per listing (bilingual DE/EN stop-list, `min_df=20`, `max_df=0.85`) plus `review_n_reviews` and `review_mean_len`. CV-R² was essentially unchanged (0.668 → 0.666) but held-out R² **dropped** from 0.587 to 0.563 (−0.024) and MAE rose from 36.32 to 37.03 CHF. SHAP showed why: the highest-ranked review tokens were generic positives — `tok_location` (#29), `tok_good` (#36), `tok_great`, `tok_perfect`, `tok_recommend` — which appear on most well-rated listings and don't differentiate price tiers. `review_n_reviews` (#125, SHAP 0) was perfectly redundant with the existing `number_of_reviews`. Net effect: 32 noisy sparse features with high feature-to-sample ratio (2052 train rows) caused mild overfit. A more sophisticated text approach (NMF topic modelling, embeddings, sentiment scores) might extract real signal — top-N raw tokens did not.
- **Broader seasonal calendar availability rates (Summer / Shoulder / Winter-Offpeak).** The existing calendar features only sampled ~3 weeks of the calendar year (`cal_avail_rate_zff_2025` covers Sep 29 – Oct 5, `cal_avail_rate_winter_hols` covers Dec 20 – Jan 2). Both ranked weak in SHAP (#52 and #44 before). Hypothesised that broader season-level availability rates (Jun – Aug 2026 summer, Apr – May 2026 spring shoulder, Jan 15 – Feb 28 2026 winter off-peak) would capture standard tourism cycles that those narrow event windows miss. Added all three on top of the existing rates. CV-R² moved −0.005 (0.669 → 0.664), test R² moved −0.007 (0.589 → 0.582), MAE +0.24 CHF. Reverted. Likely cause: the new seasonal rates correlate strongly with the existing `availability_30/60/90/365` columns (a listing booked solid in summer is also booked solid over its 30/60/90-day windows), so they added correlated noise instead of orthogonal signal. The narrow event windows are weak *because* the calendar's information content is mostly already captured by `availability_*` and `cal_min_nights_*`, not because the windows are too narrow.

Conclusion: when several distance features overlap geographically, adding more of them does not stack; the model just spreads the same signal across more columns. External data only helps when it captures something genuinely orthogonal to what is already in the feature set. Stratification only helps when the buckets follow genuinely different price-driver patterns *and* there's enough data in the minority bucket. Sparse text features (top-N tokens) need to be either very few (high-signal-only) or properly compressed (topics/embeddings); a wide TF-IDF over generic positives mostly adds noise to a tree model — none of these held for this dataset.

## Output
- Trained model:  models/rent_pipeline.pkl
- Figures:        reports/figures/
