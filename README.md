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

1. Download `listings.csv.gz` from https://insideairbnb.com/get-the-data/ (select Zurich).
2. Extract and place at `data/raw/listings.csv`.
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
        ('cat_te', TargetEncoder(cv=5, smooth='auto'),              ['neighbourhood_cleansed']),
        ('cat_oh', OneHotEncoder(handle_unknown='ignore'),          ['room_type', 'property_type']),
        ('bin',    'passthrough',                                    BINARY),
    ])),
    ('model', <winning estimator>),
])
```
A consumer can `joblib.load(...).predict(raw_dataframe)` — no separate scaler, no manual one-hot. The old `models/scaler.pkl` is obsolete and unused by the new pipeline.

## Features
- **Listing structure:** `accommodates`, `bedrooms`, `bathrooms`, `room_type`, `property_type` (31 cats), `minimum_nights`
- **Location:** `latitude`, `longitude`, **target-encoded** `neighbourhood_cleansed` (34 categories collapsed to one continuous price-prior column via out-of-fold encoding)
- **Activity:** `availability_365`, `number_of_reviews`, `reviews_per_month`
- **Event-venue proximity (6):** haversine distance in km to Hallenstadion, Letzigrund, Messe Zürich, Opernhaus, Zürich HB, plus `min_dist_venue_km`
- **Amenities (31):** `n_amenities` total count + 30 binary `amen_*` flags, picked data-driven as the 30 most-common amenity strings across all listings (nb02). Replaces an earlier hand-picked 10-flag set; the wider net let the model surface signals the curated list missed (TV, lockbox, hot-water kettle, iron).
- **Host signals (3):** `host_is_superhost`, `host_response_rate` (%), `host_acceptance_rate` (%)
- **Review scores (7):** `review_scores_rating` plus six subscores (accuracy, cleanliness, check-in, communication, location, value). All numeric; ~24% of listings have no reviews and are median-imputed inside the Pipeline.

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
| HistGradientBoosting (tuned) — target-encoded neighbourhood | —         | —          | —     | **0.641**   |
| **Winner: HistGradientBoosting** (saved to `rent_pipeline.pkl`) | **38.14** | **66.98** | **0.530** | **0.641** |

The first two rows are historical baselines (run before amenity features and tuning were added) and are kept for reference. The tuned-RF and tuned-HGB rows show only CV-R² on the training set — by policy the held-out test set is touched exactly once, on the winning model. Test metrics for the winner (MAE / RMSE / R²) are written to `reports/metrics.json` by nb05.

Cumulative lift on held-out R² from successive feature work:
- + `property_type`, `host_*`, `review_scores_*`: R² **0.478 → 0.530** (+0.052)
- + top-30 amenity one-hots (replacing hand-picked 10): R² **0.530 → 0.535** (+0.005)
- + target-encoded neighbourhood (replacing 34 one-hots): R² **0.535 → 0.530** (−0.005, kept for interpretability — see note below)

**Why keep target encoding despite the marginal R² drop?** The 34 one-hot neighbourhood columns were each contributing tiny SHAP values (top one-hot `Hard` at rank #42, ~0.0023) — the signal was real but **fragmented across 34 columns**. Target encoding consolidates the same signal into one column that lands at SHAP rank **#11** (mean |SHAP| 0.032), beating `dist_hb_km`. Predictive accuracy is essentially unchanged (within 0.005 R²), but the SHAP picture and the model's overall feature count (123 → 90) are dramatically cleaner — and the encoder's internal 5-fold CV makes it leakage-free.

Best hyperparameters from `GridSearchCV` on the current run:
- Random Forest (tuned): `n_estimators=400, max_depth=None, min_samples_leaf=1`
- HistGradientBoosting (winner): `learning_rate=0.05, max_depth=6, max_iter=400, min_samples_leaf=20`

A `DummyRegressor(strategy='mean')` baseline is also printed in nb04 to anchor what "zero skill" looks like on this data.

## Key findings (SHAP) — HistGradientBoosting winner

Mean |SHAP value| on log-price scale (top features, from `reports/figures/shap_bar.png`):

1. **`accommodates`** — strongest predictor by a wide margin (≈0.13).
2. **`bedrooms`** — clear second (≈0.08).
3. **`availability_365`** and **`property_type_Private room in rental unit`** — top-4 (≈0.05). Host-activity and granular room-type signal.
4. **`dist_opernhaus_km`** (≈0.045), **`reviews_per_month`** (≈0.04) and **`minimum_nights`** (≈0.04) — booking-dynamics and central-location signal.
5. **`review_scores_rating`** (#8), **`host_acceptance_rate`** (#9), **`room_type_Private room`** (#10) — review-quality, host-responsiveness, and the legacy 3-category room split all land within the top-10.
6. **Target-encoded `neighbourhood_cleansed` lands at #11** (≈0.032), beating `dist_hb_km` (#12). Previously this signal was spread across 34 one-hot columns whose top member (`Hard`) only reached rank #42 — encoding consolidates it into one ranked, interpretable feature.
7. **Top-30 amenity flags — what the data-driven expansion uncovered:** `amen_dishwasher` (#14), `amen_tv` (#16), `amen_lockbox` (#17) are the load-bearing amenity binaries. TV / lockbox / hot-water kettle were **not** in the original hand-picked 10 — the wider net found them. About half of the 30 amenities (kitchen, freezer, wine glasses, washer, hangers) end up below SHAP rank #60, essentially dead weight that the tree-based model simply ignores.
8. **`latitude`** sits at ~#20 (mid-pack); raw coordinates matter less once explicit distance + neighbourhood-encoded + property-type features are available.

Compared to the earliest run, the model now uses a much more diverse feature mix: structural capacity → location-by-distance → neighbourhood-encoded prior → property granularity → host quality (reviews + responsiveness) → amenities. The cumulative lift on test R² (0.478 → 0.530, with target encoding contributing 33 fewer features at the same accuracy) is consistent with these features carrying genuine pricing signal rather than just adding noise.

See `reports/figures/shap_summary.png` for the beeswarm (direction + magnitude per sample) and `shap_bar.png` for the mean-impact ranking.

## Experiments that didn't pay off

For transparency, a few feature ideas we tried but reverted because they did not improve held-out R²:

- **OSM transit stops + supermarkets in 500 m radius.** `dist_nearest_transit_km` and `n_supermarkets_500m` fetched from OpenStreetMap via the Overpass API. Zurich's transit network is so dense that almost every listing sits within ~150 m of a stop (median 112 m, max 711 m), so the distance has no discriminative power. Supermarket density correlated with `dist_hb_km` (city centre), so the model already had that signal. Both features landed at SHAP rank #26 / #30 and CV-R² dropped slightly (0.635 → 0.626).
- **Distance to Zürichsee shore + Limmat river.** Same OSM source. These features *do* carry signal (SHAP rank #12 / #13), but SHAP simply redistributed importance from the existing `dist_opernhaus_km` and `dist_hb_km` features — held-out R² changed by −0.008 (within noise). Lake/river proximity is already implicitly captured by the central-Zurich venue distances.
- **Outlier-stratified HGB (two-expert / mixture-of-experts).** Built a custom `StratifiedPriceRegressor` that trains a `HistGradientBoostingClassifier` to split listings at 250 CHF/night (~15% luxury bucket), then routes each prediction to one of two HGB regressors trained on its bucket. The stratified pipeline lost by a wide margin: CV-R² **0.558** vs. the single-HGB baseline's **0.641** (−0.083). Two reasons: the router classifier mis-routes ~10% of listings (each error costs a lot in MAE), and the luxury bucket's 299 training rows is below the sample-size regime where HGB outperforms simpler models. The single HGB sees the full distribution and learns the price-by-feature relationship without any routing risk.

Conclusion: when several distance features overlap geographically, adding more of them does not stack; the model just spreads the same signal across more columns. External data only helps when it captures something genuinely orthogonal to what is already in the feature set. Stratification only helps when the buckets follow genuinely different price-driver patterns *and* there's enough data in the minority bucket — neither held for this dataset.

## Output
- Trained model:  models/rent_pipeline.pkl
- Figures:        reports/figures/
