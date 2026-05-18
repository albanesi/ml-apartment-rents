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
- **Listing structure:** `accommodates`, `bedrooms`, `bathrooms`, `room_type`, `minimum_nights`
- **Location:** `latitude`, `longitude`, one-hot `neighbourhood_cleansed`
- **Activity:** `availability_365`, `number_of_reviews`, `reviews_per_month`
- **Event-venue proximity (6):** haversine distance in km to Hallenstadion, Letzigrund, Messe Zürich, Opernhaus, Zürich HB, plus `min_dist_venue_km`
- **Amenities (11):** `n_amenities` total count + binary flags for AC, pool, hot tub, gym, dishwasher, washer, dryer, free parking, elevator, balcony

## Results
| Model                       | MAE (CHF) | RMSE (CHF) | R²    | CV R² (log) |
|-----------------------------|-----------|------------|-------|-------------|
| Linear Regression           | 58.60     | 108.00     | 0.368 | 0.385       |
| Random Forest (n=100, untuned, baseline) | 51.38     | 100.00     | 0.458 | 0.478       |
| Random Forest (tuned)       | —         | —          | —     | 0.562       |
| HistGradientBoosting (tuned) | —         | —          | —     | **0.583**   |
| **Winner: HistGradientBoosting** (saved to `rent_pipeline.pkl`) | **41.20** | **70.61** | **0.478** | **0.583** |

The first two rows are historical baselines (run before amenity features and tuning were added) and are kept for reference. The tuned-RF and tuned-HGB rows show only CV-R² on the training set — by policy the held-out test set is touched exactly once, on the winning model. Test metrics for the winner (MAE / RMSE / R²) are written to `reports/metrics.json` by nb05.

Best hyperparameters from `GridSearchCV` on the current run:
- Random Forest (tuned): `n_estimators=400, max_depth=None, min_samples_leaf=1`
- HistGradientBoosting (winner): `learning_rate=0.1, max_depth=6, max_iter=200, min_samples_leaf=10`

A `DummyRegressor(strategy='mean')` baseline is also printed in nb04 to anchor what "zero skill" looks like on this data.

## Key findings (SHAP) — HistGradientBoosting winner

Mean |SHAP value| on log-price scale (top features, from `reports/figures/shap_bar.png`):

1. **`accommodates`** — strongest predictor by a wide margin (≈0.14), now overtaking bedrooms.
2. **`bedrooms`** and **`room_type_Private room`** — tied for second (≈0.07). Private-room listings consistently push the predicted price down.
3. **`dist_hb_km`** (distance to Zürich HB) — top-4 feature; the venue-proximity features are pulling their weight, with `dist_opernhaus_km` and `dist_letzigrund_km` also in the upper half of the ranking.
4. **Host-activity signals** — `availability_365` and `reviews_per_month` — are surprisingly important (top-6), suggesting how actively a listing is run carries real price information.
5. **`has_dishwasher`** is the strongest of the amenity binary flags (top-9). The other `has_*` flags contribute much less.
6. **`latitude`** dropped from top-3 (old RF baseline) to mid-pack (~#11). With explicit venue-distance features available, the model relies less on raw geographic coordinates.

Compared to the previous Random Forest baseline (which had bedrooms #1, accommodates #2, latitude #3), the new HGB model uses a more diverse feature mix: structural capacity → location-by-distance → host-activity → amenities.

See `reports/figures/shap_summary.png` for the beeswarm (direction + magnitude per sample) and `shap_bar.png` for the mean-impact ranking.

## Output
- Trained model:  models/rent_pipeline.pkl
- Figures:        reports/figures/
