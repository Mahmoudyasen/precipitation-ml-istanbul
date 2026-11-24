# Machine Learning Techniques to Predict Total Precipitation: Istanbul

Code and data for the conference study **"Machine Learning Techniques to Predict Total Precipitation"**.
Random Forest and Gradient Boosting models predict hourly total precipitation over Istanbul.
They use ERA5-Land reanalysis data and MODIS precipitable water vapour (TPW).

**Authors:** Omar Sawalha, Mahmoud Yaseen
**Supervisor:** Prof. Zafer Aslan, Istanbul Aydın University

---

## Study area & data

| | |
|---|---|
| Region | Istanbul, 3 × 3 ERA5-Land grid (lat 40.85 / 41.10 / 41.35 °N, lon 28.75 / 29.00 / 29.25 °E) |
| Period | 2021-01-01 → 2024-12-31, hourly (35,064 time steps × 9 cells) |
| ERA5-Land | `tp`: total precipitation (m) · `t2m`: 2 m temperature (K) |
| MODIS (Terra) | `ir_terra`, `nir_terra`, `modis_ir_cm`, `modis_nir_cm`: precipitable water vapour (cm), IR and NIR retrievals |
| Split | train 2021–2023 · test 2024 |

ERA5-Land source: [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/), ERA5-Land hourly.

> **Note:** The input dataset (ERA5 merged with MODIS TPW, `combined_with_modisf_tpw.csv`) is **not included yet** and will be added later.
> The model expects an hourly CSV with the columns `valid_time, latitude, longitude, tp, t2m_C`, plus optionally the MODIS columns above.
> Without the MODIS columns, it runs on ERA5 features alone and prints a warning.
> In that case the hourly and seasonal scores come out almost the same (for example, hourly Gradient Boosting R = 0.841). The daily scores drop.

## Method

1. **Preprocessing:** start from an hourly time × lat × lon table of ERA5 values with t2m in °C, joined with MODIS TPW. Fill missing MODIS values with the training-set mean.
2. **Features:**
   - calendar: hour, day of week, month, day of year
   - grid-cell location
   - MODIS TPW
   - leakage-safe lags and rolling means of `tp` and `t2m_C`, per grid cell: lag 1 and 2, rolling 3 and 7, each built from `shift(1)`
3. **Models:**
   - `RandomForestRegressor(n_estimators=300)`
   - `GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)`
4. **Scenarios:**
   - hourly, all year
   - daily: `tp` summed per day, other variables averaged
   - one hourly model per season (DJF / MAM / JJA / SON)

## Results (test year 2024)

From [`notebooks/03_final_predictions.ipynb`](notebooks/03_final_predictions.ipynb).
The slides report **R = √R²**.

| Scenario | Model | MSE (m²) | R² | R |
|---|---|---|---|---|
| Hourly, all year | Random Forest | 2.41 × 10⁻⁸ | 0.697 | 0.835 |
| Hourly, all year | **Gradient Boosting** | **2.33 × 10⁻⁸** | **0.708** | **0.841** |
| Winter (DJF) | **Gradient Boosting** | 1.36 × 10⁻⁸ | 0.811 | **0.901** |
| Spring (MAM) | **Gradient Boosting** | 1.53 × 10⁻⁸ | 0.727 | **0.853** |
| Summer (JJA) | **Gradient Boosting** | 1.62 × 10⁻⁸ | 0.440 | **0.663** |
| Autumn (SON) | **Random Forest** | 4.95 × 10⁻⁸ | 0.688 | **0.829** |
| Daily | Gradient Boosting | 1.32 × 10⁻⁵ | 0.132 | 0.364 |

**Takeaways:**
- Gradient Boosting is best overall and in winter, spring and summer. Random Forest is slightly better in autumn.
- Summer convective rainfall is the hardest to predict.
- Daily aggregation scores much lower than the hourly models.

## Repository layout

```
notebooks/03_final_predictions.ipynb   final pipeline, with saved outputs (paper results)
src/train_models.py                    script version (writes metrics and predictions CSVs)
results/                               output folder
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/train_models.py --csv data/combined_with_modisf_tpw.csv --out-dir results
```

## Future work

- Add more predictors: wind, humidity, CAPE.
- Try sequence models: LSTM, Transformers.
- Build a real-time prediction system.
