# Predicting Apartment Rents — Group 2

**Course:** Fundamentals of Python & Applications in Data Science
**Option:** B â€” Machine Learning

## Group members
<!-- Add your names before submission -->
- Abel ...
- ...

## Setup
pip install -r requirements.txt

## Data
**Source:** Inside Airbnb â€” Zurich, detailed listings file (79 columns)
**Snapshot date:** 2025-09-29 (scrape_id 20250929042214)
**Rows:** 3 417 listings

Processed train/test splits are committed under `data/processed/` so the pipeline
runs without re-downloading. To reproduce from the raw file:

1. Download `listings.csv.gz` from https://insideairbnb.com/get-the-data/ (select Zurich).
2. Extract and place at `data/raw/listings.csv`.
3. Run notebooks 01 â€” 05 in order.

IMPORTANT: Use the detailed version (~79 columns). The simple 18-column file will break notebook 02.

## Reproduction — run notebooks in this exact order
1. 01_data_ingestion.ipynb  — loads raw data, saves to data/processed/
2. 02_eda.ipynb             — EDA plots, saves updated data/processed/listings_eda.csv
3. 03_preprocessing.ipynb  — cleaning, encoding, splitting, saves X_train/X_test/y_train/y_test
4. 04_modeling.ipynb        — trains models, saves models/rent_pipeline.pkl
5. 05_evaluation_xai.ipynb — metrics, residuals, SHAP plots

Each notebook depends on the previous one. Always restart kernel and run all cells.

## Results
| Model             | MAE (CHF) | RMSE (CHF) | R²    |
|-------------------|-----------|------------|-------|
| Linear Regression | 77.40     | 117.30     | 0.254 |
| Random Forest     | TBD*      | TBD*       | TBD*  |



*Re-run notebooks 03â€“05 after log-transforming the target to get updated metrics.

## Key findings (SHAP)
- Bedrooms is the strongest price predictor (mean SHAP ~30)
- Accommodates and latitude follow as second and third
- Private room type reduces predicted price significantly
- Neighbourhood effects are present but individually small
- Remaining variance reflects host discretion not captured by structured features

## Output
- Trained model:  models/rent_pipeline.pkl
- Figures:        reports/figures/
