"""Train Random Forest and Gradient Boosting models to predict ERA5-Land total precipitation (tp).

Script version of notebooks/03_final_predictions.ipynb, the pipeline behind the
conference results. Plots are left out; metrics and predictions are written as CSVs.

Scenarios:
  1) Hourly: train on < 2024, test on 2024
  2) Daily: per grid cell, tp is summed and continuous variables are averaged
  3) Seasonal (hourly data): one model per season (DJF, MAM, JJA, SON)

Lag and rolling features are built leakage-free with shift(1), per grid cell.

Usage:
    python src/train_models.py --csv data/combined_with_modisf_tpw.csv --out-dir results
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42
TEST_START = pd.Timestamp("2024-01-01")

CANDIDATE_FEATURES = [
    "latitude", "longitude",
    "ir_terra", "nir_terra", "modis_ir_cm", "modis_nir_cm",
    "hour", "dayofweek", "month", "dayofyear",
    "t2m_C_lag1", "t2m_C_lag2", "t2m_C_roll3", "t2m_C_roll7",
    "tp_lag1", "tp_lag2", "tp_roll3", "tp_roll7",
]
MODIS_COLS = ["ir_terra", "nir_terra", "modis_ir_cm", "modis_nir_cm"]
SEASON_MONTHS = {"Winter": [12, 1, 2], "Spring": [3, 4, 5], "Summer": [6, 7, 8], "Autumn": [9, 10, 11]}


def make_models():
    return {
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                       random_state=RANDOM_STATE),
    }


def add_calendar(df, time_col="valid_time"):
    df["hour"] = df[time_col].dt.hour
    df["dayofweek"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
    df["dayofyear"] = df[time_col].dt.dayofyear
    return df


def add_lags_rolls(df, cols, group_keys=("latitude", "longitude"), lags=(1, 2), rolls=(3, 7)):
    df = df.copy()
    g = df.groupby(list(group_keys))
    for col in cols:
        if col not in df.columns:
            continue
        for L in lags:
            df[f"{col}_lag{L}"] = g[col].shift(L)
        for W in rolls:
            df[f"{col}_roll{W}"] = g[col].transform(lambda s: s.shift(1).rolling(W, min_periods=1).mean())
    return df


def fit_predict(X_train, y_train, X_test, y_test):
    imp = SimpleImputer(strategy="mean")
    Xtr, Xte = imp.fit_transform(X_train), imp.transform(X_test)
    preds, metrics = {}, {}
    for name, model in make_models().items():
        model.fit(Xtr, y_train)
        y_hat = model.predict(Xte)
        preds[name] = y_hat
        metrics[name] = {"MSE": mean_squared_error(y_test, y_hat), "R2": r2_score(y_test, y_hat)}
    return preds, metrics


def run_scenario(df, scenario, subset, rows):
    feats = [c for c in CANDIDATE_FEATURES if c in df.columns]
    tr, te = df[df["valid_time"] < TEST_START], df[df["valid_time"] >= TEST_START]
    if tr.empty or te.empty:
        print(f"[{scenario}/{subset}] not enough rows for split; skipped")
        return None
    preds, metrics = fit_predict(tr[feats], tr["tp"], te[feats], te["tp"])
    for m, v in metrics.items():
        rows.append({"Scenario": scenario, "Subset": subset, "Model": m, **v})
    out = te[["valid_time", "latitude", "longitude", "tp"]].rename(columns={"tp": "actual"}).copy()
    for m, y in preds.items():
        out[m] = y
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/combined_with_modisf_tpw.csv",
                   help="hourly ERA5 (+ optional MODIS TPW) table")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.csv)
    data["valid_time"] = pd.to_datetime(data["valid_time"], errors="coerce")
    data = data.dropna(subset=["valid_time"]).sort_values(["valid_time", "latitude", "longitude"])
    if not any(c in data.columns for c in MODIS_COLS):
        print("[Warning] no MODIS TPW columns found: training on ERA5 features only "
              "(results will differ from the paper)")

    data = add_calendar(data)
    data = add_lags_rolls(data, ["tp", "t2m_C"]).dropna(subset=["tp"])

    rows = []
    # 1) Hourly
    pred_hr = run_scenario(data, "Hourly", "Test_2024plus", rows)

    # 2) Daily
    d = data.assign(date=data["valid_time"].dt.floor("D"))
    agg = {"tp": "sum", **{c: "mean" for c in ["t2m_C", *MODIS_COLS] if c in d.columns}}
    d = d.groupby(["date", "latitude", "longitude"], as_index=False).agg(agg).rename(columns={"date": "valid_time"})
    d = add_calendar(d).sort_values(["valid_time", "latitude", "longitude"])
    d = add_lags_rolls(d, ["tp", "t2m_C"]).dropna(subset=["tp"])
    pred_d = run_scenario(d, "Daily", "Test_2024plus", rows)

    # 3) Seasonal (hourly)
    seasonal = []
    for season, months in SEASON_MONTHS.items():
        s = data[data["month"].isin(months)].sort_values(["valid_time", "latitude", "longitude"])
        s = add_lags_rolls(s, ["tp", "t2m_C"])
        out = run_scenario(s, "Seasonal(Hourly)", f"{season}_Test_2024plus", rows)
        if out is not None:
            seasonal.append(out.assign(Season=season))

    metrics = pd.DataFrame(rows)
    metrics["R"] = metrics["R2"].clip(lower=0) ** 0.5  # the slides/paper report R = sqrt(R^2)
    print(metrics.to_string(index=False))

    metrics.to_csv(out_dir / "metrics_overview.csv", index=False)
    if pred_hr is not None:
        pred_hr.to_csv(out_dir / "predictions_hourly_test_2024plus.csv", index=False)
    if pred_d is not None:
        pred_d.to_csv(out_dir / "predictions_daily_test_2024plus.csv", index=False)
    if seasonal:
        pd.concat(seasonal, ignore_index=True).to_csv(out_dir / "predictions_seasonal_hourly_test_2024plus.csv",
                                                       index=False)
    print(f"[Saved] metrics and predictions to {out_dir}/")


if __name__ == "__main__":
    main()
