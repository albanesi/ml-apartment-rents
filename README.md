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
        ('cat_te', TargetEncoder(cv=5, smooth='auto'),              ['neighbourhood_cleansed']),
        ('cat_oh', OneHotEncoder(handle_unknown='ignore'),          ['room_type', 'property_type']),
        ('bin',    'passthrough',                                    BINARY),
    ])),
    ('model', <winning estimator>),
])
```
A consumer can `joblib.load(...).predict(raw_dataframe)` — no separate scaler, no manual one-hot. The old `models/scaler.pkl` is obsolete and unused by the new pipeline.

## Features
- **Listing structure:** `accommodates`, `bedrooms`, `bathrooms`, `beds`, `room_type`, `property_type` (31 cats), `minimum_nights`
- **Location:** `latitude`, `longitude`, **target-encoded** `neighbourhood_cleansed` (34 categories collapsed to one continuous price-prior column via out-of-fold encoding)
- **Booking dynamics:** `availability_30`, `availability_60`, `availability_90`, `availability_365` (short / medium / long-term availability), `instant_bookable` (binary)
- **Review activity:** `number_of_reviews`, `number_of_reviews_ltm` (last 12 months), `number_of_reviews_l30d` (last 30 days), `reviews_per_month`
- **Event-venue proximity (6):** haversine distance in km to Hallenstadion, Letzigrund, Messe Zürich, Opernhaus, Zürich HB, plus `min_dist_venue_km`
- **Amenities (31):** `n_amenities` total count + 30 binary `amen_*` flags, picked data-driven as the 30 most-common amenity strings across all listings (nb02).
- **Host signals (5):** `host_is_superhost`, `host_response_rate` (%), `host_acceptance_rate` (%), `host_response_time` (ordinal 1–4, lower = faster), `host_total_listings_count` (signals professional vs. private hosts)
- **Review scores (7):** `review_scores_rating` plus six subscores (accuracy, cleanliness, check-in, communication, location, value). All numeric; ~24% of listings have no reviews and are median-imputed inside the Pipeline.
- **Text features (14)** from `name` + `description`: 2 length features (`text_name_len`, `text_desc_len`) + 12 bilingual keyword flags (`text_luxury`, `text_view`, `text_central`, `text_modern`, `text_terrace`, `text_renovated`, `text_lake`, `text_river`, `text_quiet`, `text_design`, `text_spacious`, `text_charming`) — regexes match both German and English variants.
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
| HistGradientBoosting (tuned) — + calendar.csv features | —         | —          | —     | **0.668**   |
| **Winner: HistGradientBoosting** (saved to `rent_pipeline.pkl`) | **36.32** | **62.81** | **0.587** | **0.668** |

The first two rows are historical baselines (run before amenity features and tuning were added) and are kept for reference. The tuned-RF and tuned-HGB rows show only CV-R² on the training set — by policy the held-out test set is touched exactly once, on the winning model. Test metrics for the winner (MAE / RMSE / R²) are written to `reports/metrics.json` by nb05.

Cumulative lift on held-out R² from successive feature work:
- + `property_type`, `host_*`, `review_scores_*`: R² **0.478 → 0.530** (+0.052)
- + top-30 amenity one-hots (replacing hand-picked 10): R² **0.530 → 0.535** (+0.005)
- + target-encoded neighbourhood (replacing 34 one-hots): R² **0.535 → 0.530** (−0.005, kept for interpretability — see note below)
- + expanded columns from the 79-col file (`host_total_listings_count`, `beds`, `instant_bookable`, `availability_30/60/90`, `number_of_reviews_ltm/l30d`, `host_response_time`) + 14 text features from `name`/`description`: R² **0.530 → 0.573** (+0.043)
- + 6 calendar-derived features (min/max-stay strategy, weekend pressure, ZFF + Xmas availability): R² **0.573 → 0.587** (+0.014)

**Why keep target encoding despite the marginal R² drop?** The 34 one-hot neighbourhood columns were each contributing tiny SHAP values (top one-hot `Hard` at rank #42, ~0.0023) — the signal was real but **fragmented across 34 columns**. Target encoding consolidates the same signal into one column that lands in the SHAP top-10, beating `dist_hb_km`. Predictive accuracy is essentially unchanged at that step, but the SHAP picture and the model's overall feature count are dramatically cleaner — and the encoder's internal 5-fold CV makes it leakage-free.

Best hyperparameters from `GridSearchCV` on the current run:
- Random Forest (tuned): `n_estimators=200, max_depth=30, min_samples_leaf=1`
- HistGradientBoosting (winner): `learning_rate=0.05, max_depth=None, max_iter=400, min_samples_leaf=20`

A `DummyRegressor(strategy='mean')` baseline is also printed in nb04 to anchor what "zero skill" looks like on this data.

## Key findings (SHAP) — HistGradientBoosting winner

Mean |SHAP value| on log-price scale (top features, from `reports/figures/shap_bar.png`):

1. **`accommodates`** (≈0.12) and **`bedrooms`** (≈0.08) — structural capacity dominates.
2. **`property_type_Private room in rental unit`** (#3, ≈0.052) and **`room_type_Private room`** (#5, ≈0.042) — granular room split.
3. **`host_total_listings_count` at #4** (≈0.048) — distinguishes professional managers (max 2 396 listings in the data!) from private hosts.
4. **`availability_30`** (#6, ≈0.035), **`neighbourhood_cleansed`** target-encoded (#7, ≈0.035), **`dist_opernhaus_km`** (#8), **`dist_hb_km`** (#9) — booking pressure and central-location signal.
5. **Calendar features land high: `cal_min_nights_median` at #10** (≈0.028) — the typical minimum-stay setting from the per-day calendar carries more signal than the single-value `minimum_nights` snapshot (which sits at #25). `cal_avail_weekend_premium` (#22) and `cal_min_nights_std` (#23) also land top-25; `cal_max_nights_median` is #26.
6. **`review_scores_rating`** (#11), **`host_acceptance_rate`** (#12), **`amen_tv`** (#13), **`availability_90`** (#14) — host-quality and booking-window signals.
7. **`amen_dishwasher`** (#16), **`number_of_reviews_l30d`** (#19), **`beds`** (#20) — amenity + booking-velocity + structural follow-ups. About half of the 30 amenities sit below SHAP rank #60 — dead weight the tree model ignores.
8. **`text_modern`** at ~#28 still represents the text-keyword family. "Modern"/"moderne"/"stylish" in the listing copy carries a real price premium; `text_desc_len` and `text_name_len` are nearby as quality proxies.
9. **Surprising flops:** `instant_bookable` and several text keywords (`text_view`, `text_lake`, `text_river`, `text_renovated`) sit below rank #60. Also flat: `cal_avail_rate_zff_2025` (#52) and `cal_avail_rate_winter_hols` (#44) — event-period availability is less differentiating than the full-year strategy aggregates, probably because too few days are in each window.

Compared to the earliest run, the model now uses a much more diverse feature mix: structural capacity → location (distance + neighbourhood prior) → property granularity → host professionalisation → booking dynamics (incl. per-day calendar strategy) → review quality → amenities → text framing. The cumulative lift on test R² (0.478 → 0.587, +0.109) is consistent with each of these feature families carrying genuine, partially-orthogonal pricing signal.

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
