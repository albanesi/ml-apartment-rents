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
        ('num', SimpleImputer(median) → StandardScaler, NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore'),  ['neighbourhood_cleansed', 'room_type']),
        ('bin', 'passthrough',                            has_* flags),
    ])),
    ('model', <winning estimator>),
])
```
A consumer can `joblib.load(...).predict(raw_dataframe)` — no separate scaler, no manual one-hot. The old `models/scaler.pkl` is obsolete and unused by the new pipeline.

## Features
- **Listing structure:** `accommodates`, `bedrooms`, `bathrooms`, `room_type`, `property_type` (31 cats), `minimum_nights`
- **Location:** `latitude`, `longitude`, one-hot `neighbourhood_cleansed`
- **Activity:** `availability_365`, `number_of_reviews`, `reviews_per_month`
- **Event-venue proximity (6):** haversine distance in km to Hallenstadion, Letzigrund, Messe Zürich, Opernhaus, Zürich HB, plus `min_dist_venue_km`
- **Amenities (11):** `n_amenities` total count + binary flags for AC, pool, hot tub, gym, dishwasher, washer, dryer, free parking, elevator, balcony
- **Host signals (3):** `host_is_superhost`, `host_response_rate` (%), `host_acceptance_rate` (%)
- **Review scores (7):** `review_scores_rating` plus six subscores (accuracy, cleanliness, check-in, communication, location, value). All numeric; ~24% of listings have no reviews and are median-imputed inside the Pipeline.

## Results
| Model                       | MAE (CHF) | RMSE (CHF) | R²    | CV R² (log) |
|-----------------------------|-----------|------------|-------|-------------|
| Linear Regression           | 58.60     | 108.00     | 0.368 | 0.385       |
| Random Forest (n=100, untuned, baseline) | 51.38     | 100.00     | 0.458 | 0.478       |
| Random Forest (tuned)       | —         | —          | —     | 0.583       |
| HistGradientBoosting (tuned) | —         | —          | —     | **0.635**   |
| **Winner: HistGradientBoosting** (saved to `rent_pipeline.pkl`) | **38.80** | **67.02** | **0.530** | **0.635** |

The first two rows are historical baselines (run before amenity features and tuning were added) and are kept for reference. The tuned-RF and tuned-HGB rows show only CV-R² on the training set — by policy the held-out test set is touched exactly once, on the winning model. Test metrics for the winner (MAE / RMSE / R²) are written to `reports/metrics.json` by nb05.

Adding `property_type`, `host_*`, and `review_scores_*` features (this PR) lifted held-out R² from 0.478 → 0.530 (+0.052) and dropped MAE from 41.20 → 38.80 CHF on the same train/test split.

Best hyperparameters from `GridSearchCV` on the current run:
- Random Forest (tuned): `n_estimators=400, max_depth=30, min_samples_leaf=3`
- HistGradientBoosting (winner): `learning_rate=0.05, max_depth=None, max_iter=400, min_samples_leaf=10`

A `DummyRegressor(strategy='mean')` baseline is also printed in nb04 to anchor what "zero skill" looks like on this data.

## Key findings (SHAP) — HistGradientBoosting winner

Mean |SHAP value| on log-price scale (top features, from `reports/figures/shap_bar.png`):

1. **`accommodates`** — strongest predictor by a wide margin (≈0.13).
2. **`bedrooms`** — clear second (≈0.07).
3. **`availability_365`** and **`dist_opernhaus_km`** — top-4 (≈0.05). Host-activity and central-location signal.
4. **`dist_hb_km`** and the new **`property_type_Private room in rental unit`** (≈0.04) — the 31-category `property_type` captures finer distinctions than the 3-category `room_type` and overtakes it in importance.
5. **New review/host features land in the top-10:** `review_scores_rating` (#8) and `host_acceptance_rate` (#9). `review_scores_cleanliness` is also in the top-15. These features came in fresh in this PR and are now load-bearing — `review_scores_rating` alone outranks `n_amenities` and every individual `has_*` flag.
6. **`room_type_Private room`** dropped from tied-second to #10 — its signal is now shared with the more specific `property_type` categories.
7. **`latitude`** sits at ~#12 (mid-pack); raw coordinates matter less once explicit distance + property-type features are available.

Compared to the previous run without host/review/property_type features, the model uses a much more diverse feature mix: structural capacity → location-by-distance → property granularity → host quality (reviews + responsiveness) → amenities. The lift on test R² (0.478 → 0.530) is consistent with these features carrying genuine pricing signal rather than just adding noise.

See `reports/figures/shap_summary.png` for the beeswarm (direction + magnitude per sample) and `shap_bar.png` for the mean-impact ranking.

## Output
- Trained model:  models/rent_pipeline.pkl
- Figures:        reports/figures/
