#
#
# This script implements a miniature research project centered on the
# forecasting of implied volatility (IV) and an earnings event study.  In
# practice, single‑stock volatility indices (e.g. VXAPL for Apple) are
# proprietary and difficult to obtain without subscription.  To
# demonstrate the methodology in a reproducible way using publicly
# available data, this script uses the CBOE VIX index as a proxy for IV.
#
# The workflow follows the structure described in a coaching email from
# Quant Blueprint and encapsulates the following steps:
#
# 1. **Data Loading:** Read a CSV containing daily VIX close values
#    (provided locally in the repository).  Parse dates, drop missing
#    entries and restrict to a reasonable date range for the study.
#
# 2. **Feature Engineering:** Construct lagged versions of the IV series,
#    differences, a simple realised volatility proxy (20‑day rolling
#    standard deviation of the daily change) and weekday dummy
#    variables.  These features mirror the exogenous signals one might
#    include in an ARIMAX model.
#
# 3. **Baselines:** Compute naive and EWMA forecasts to establish
#    reference error levels.  The EWMA smoothing parameter is chosen
#    conservatively.
#
# 4. **Time‑Aware Cross‑Validation:** Implement a rolling/expanding
#    window strategy where the training set grows over time and
#    validation is performed on successive calendar years.  Within each
#    fold, one‑step‑ahead forecasts are generated.
#
# 5. **Models:** Fit ARIMA and ARIMAX models using `pmdarima`.
#    Auto‑selection is used to find suitable (p,d,q) orders within
#    constrained ranges.  The ARIMAX model leverages the engineered
#    exogenous features.  A simple realised volatility proxy stands in
#    for the unavailable GARCH(1,1) variance forecast.
#
# 6. **Evaluation:** For each fold and model, compute root mean square
#    error (RMSE), mean absolute error (MAE) and a direction hit rate
#    (proportion of correctly predicted signs of daily changes).  A
#    comparison table summarises average performance across folds.
#
# 7. **Event Study:** Using a list of Apple earnings dates sourced from
#    public earnings calendars (also included in this repository),
#    quantify the typical behaviour of VIX around earnings announcements.
#    A ramp (pre‑event) and crush (post‑event) metric is calculated
#    across events, and simple statistical tests (t‑tests and signed‑rank
#    tests) assess whether the average changes differ from zero.
#
# The goal of this script is not to produce a perfect volatility
# forecast, but to demonstrate a clean, leakage‑free workflow that
# mirrors what would be done with proprietary single‑stock IV data.  The
# analysis produces tables and figures that can be referenced in a
# resume or interview discussion.
# """

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats


@dataclass
class FoldResult:
    """Container for storing results from a single CV fold."""
    fold_name: str
    model_name: str
    rmse: float
    mae: float
    direction_hit: float


def load_vix_data(csv_path: str, start_date: str = "2010-01-01") -> pd.Series:
    """Load VIX close values from a CSV file and return a time series.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file with columns `observation_date` and `VIXCLS`.
    start_date : str
        The earliest date to include in the returned series.

    Returns
    -------
    pd.Series
        Series indexed by date with the VIX close values as floats.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "observation_date" not in df.columns or "VIXCLS" not in df.columns:
        raise ValueError("Expected columns 'observation_date' and 'VIXCLS' in VIX data")
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.set_index("observation_date").sort_index()
    # Filter to start_date onwards
    ts = df.loc[df.index >= pd.to_datetime(start_date), "VIXCLS"].astype(float)
    ts = ts.dropna()
    return ts


def engineer_features(vix: pd.Series) -> pd.DataFrame:
    """Construct feature matrix from the VIX series.

    This includes lagged levels (1, 5 and 10 days), daily differences,
    a 20‑day rolling standard deviation of the differences as a realised
    volatility proxy, and weekday dummies.

    Parameters
    ----------
    vix : pd.Series
        Daily VIX close values indexed by date.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: `iv` (target), `iv_lag1`, `iv_lag5`,
        `iv_lag10`, `iv_diff`, `iv_roll_std20`, `weekday_mon`, ... `weekday_fri`.
    """
    df = pd.DataFrame({"iv": vix})
    # Lagged levels
    df["iv_lag1"] = df["iv"].shift(1)
    df["iv_lag5"] = df["iv"].shift(5)
    df["iv_lag10"] = df["iv"].shift(10)
    # First difference (daily change)
    df["iv_diff"] = df["iv"].diff(1)
    # Realised volatility proxy: rolling std of daily change
    df["iv_roll_std20"] = df["iv_diff"].rolling(window=20, min_periods=10).std()
    # Weekday dummies
    df["weekday"] = df.index.dayofweek
    weekdays = pd.get_dummies(df["weekday"], prefix="weekday", drop_first=True)
    df = pd.concat([df, weekdays], axis=1)
    df = df.drop(columns=["weekday"])
    # Drop initial rows with NaN features
    df = df.dropna()
    return df


def compute_baselines(y: pd.Series) -> Dict[str, np.ndarray]:
    """Compute naive and EWMA forecasts for a series.

    Parameters
    ----------
    y : pd.Series
        Target series.

    Returns
    -------
    dict of str -> np.ndarray
        Dictionary mapping baseline names to forecast arrays
        aligned to the index of `y` (excluding the first value).
    """
    # Align predictions with y index (forecast for time t uses info up to t-1)
    naive_pred = y.shift(1)
    # EWMA smoothing parameter (lambda); choose 0.94 to emphasise recent days
    lam = 0.94
    ewma_pred = y.ewm(alpha=1 - lam, adjust=False).mean().shift(1)
    return {
        "naive": naive_pred,
        "ewma": ewma_pred,
    }


def rolling_cross_validation(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    date_index: pd.DatetimeIndex,
    fold_years: List[Tuple[int, int]],
    max_arima_p: int = 2,
    max_arima_q: int = 2,
    max_arima_d: int = 1,
) -> List[FoldResult]:
    """Perform rolling/expanding cross‑validation using ARIMA/ARIMAX models.

    Each fold is defined by a tuple (train_end_year, test_year).  The
    training set includes all observations with year <= train_end_year;
    the validation set includes all observations from January 1 of
    test_year until December 31 of test_year.  One‑step forecasts are
    generated for each day in the validation set.

    Parameters
    ----------
    data : pd.DataFrame
        Feature DataFrame with target and exogenous variables.  Must
        be indexed by date.
    target_col : str
        Column name of the target variable.
    feature_cols : list of str
        List of exogenous feature column names.
    date_index : pd.DatetimeIndex
        Index corresponding to `data`.  Typically `data.index`.
    fold_years : list of (int, int)
        List of tuples (train_end_year, test_year).  For each
        (te, ty), training data are those with year <= te, test data
        those with year == ty.
    max_arima_p, max_arima_q, max_arima_d : int
        Maximum p, q, d orders for the auto_arima search.

    Returns
    -------
    list of FoldResult
        Performance metrics for each model and fold.
    """
    results: List[FoldResult] = []
    target = data[target_col]
    exog = data[feature_cols]
    for train_end_year, test_year in fold_years:
        # Define train and test indices
        train_mask = date_index.year <= train_end_year
        test_mask = date_index.year == test_year
        y_train = target.loc[train_mask]
        y_test = target.loc[test_mask]
        X_train = exog.loc[train_mask]
        X_test = exog.loc[test_mask]
        if len(y_train) == 0 or len(y_test) == 0:
            continue
        fold_name = f"train<= {train_end_year}, test= {test_year}"
        # Baselines
        baselines = compute_baselines(y_train.append(y_test))
        for baseline_name, baseline_full in baselines.items():
            # Align baseline predictions with test period
            pred = baseline_full.loc[y_test.index]
            # Evaluate metrics
            err = pred - y_test
            rmse = np.sqrt(np.nanmean(err ** 2))
            mae = np.nanmean(np.abs(err))
            # Direction hit rate
            actual_change = y_test.diff().dropna()
            pred_change = pred.diff().dropna()
            hits = (np.sign(actual_change) == np.sign(pred_change)).astype(int)
            direction_hit = hits.mean() if len(hits) > 0 else np.nan
            results.append(FoldResult(fold_name, f"baseline_{baseline_name}", rmse, mae, direction_hit))
        # ARIMA (univariate)
        # Use a fixed ARIMA order to reduce runtime.  Empirically a (1,1,1) model
        # captures short‑memory behaviour in VIX reasonably well and avoids the
        # overhead of auto_arima.
        order = (1, 1, 1)
        # Fit ARIMA via SARIMAX for forecasting convenience
        arima_fit = SARIMAX(y_train, order=order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        # Perform one‑step forecasting on test set using dynamic updating
        preds_arima: List[float] = []
        arima_state = arima_fit
        for dt in y_test.index:
            # Forecast next value: returns a Series with arbitrary index
            fc_series = arima_state.forecast(steps=1)
            # Extract scalar forecast
            forecast = fc_series.iloc[0]
            preds_arima.append(float(forecast))
            # Update model with the actual observation to roll the state forward
            arima_state = arima_state.append([y_test.loc[dt]], refit=False)
        pred_arima = pd.Series(preds_arima, index=y_test.index)
        err = pred_arima - y_test
        rmse = np.sqrt(np.nanmean(err ** 2))
        mae = np.nanmean(np.abs(err))
        actual_change = y_test.diff().dropna()
        pred_change = pred_arima.diff().dropna()
        hits = (np.sign(actual_change) == np.sign(pred_change)).astype(int)
        direction_hit = hits.mean() if len(hits) > 0 else np.nan
        results.append(FoldResult(fold_name, f"ARIMA{order}", rmse, mae, direction_hit))
        # ARIMAX (exogenous). Fit auto order on a subset for efficiency
        # Use a fixed ARIMAX order to reduce runtime.
        order_x = (1, 1, 1)
        # Fit ARIMAX once on the full training set
        arimax_fit = SARIMAX(y_train, exog=X_train, order=order_x, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        # Forecast for the test period using provided exogenous values.
        # We use dynamic=False so that predictions are generated recursively using the observed history.
        try:
            pred_arimax_arr = arimax_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test)
        except Exception:
            # Some versions require get_prediction
            pred_arimax_arr = arimax_fit.get_prediction(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test).predicted_mean
        pred_arimax = pd.Series(pred_arimax_arr.values, index=y_test.index)
        err = pred_arimax - y_test
        rmse = np.sqrt(np.nanmean(err ** 2))
        mae = np.nanmean(np.abs(err))
        actual_change = y_test.diff().dropna()
        pred_change = pred_arimax.diff().dropna()
        hits = (np.sign(actual_change) == np.sign(pred_change)).astype(int)
        direction_hit = hits.mean() if len(hits) > 0 else np.nan
        results.append(FoldResult(fold_name, f"ARIMAX{order_x}", rmse, mae, direction_hit))
    return results


def perform_event_study(vix_series: pd.Series, event_dates: List[pd.Timestamp], pre_window: int = 10, post_window: int = 2) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Conduct a simple event study on the VIX around earnings events.

    For each event date, compute the average VIX level in the pre‑event
    window (E‑pre_window .. E‑1) and the average level in the post‑event
    window (E .. E+post_window).  Ramp = pre‑event mean minus the value
    at the start of the window; Crush = post‑event mean minus the
    value at E‑1.

    Parameters
    ----------
    vix_series : pd.Series
        Series of daily VIX values indexed by date.
    event_dates : list of pd.Timestamp
        Dates of earnings events.
    pre_window : int
        Number of business days before the event to include in the
        pre‑event window.
    post_window : int
        Number of business days after the event to include in the
        post‑event window.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        DataFrame with event‑level ramp and crush values and summary
        statistics (mean ramp/crush, t‑test p‑values, Wilcoxon p‑values).
    """
    ramps = []
    crushes = []
    valid_events = []
    for ed in event_dates:
        if ed not in vix_series.index:
            continue
        # Determine pre and post windows (business days)
        pre_start = vix_series.index.get_loc(ed) - pre_window
        pre_end = vix_series.index.get_loc(ed) - 1
        post_start = vix_series.index.get_loc(ed)
        post_end = vix_series.index.get_loc(ed) + post_window
        if pre_start < 0 or post_end >= len(vix_series):
            continue  # skip events near edges
        pre_vals = vix_series.iloc[pre_start : pre_end + 1]
        post_vals = vix_series.iloc[post_start : post_end + 1]
        # Ramp: average of pre‑window minus value at pre_start
        baseline_pre = vix_series.iloc[pre_start]
        ramp = pre_vals.mean() - baseline_pre
        # Crush: average of post‑window minus value one day before event
        baseline_post = vix_series.iloc[pre_end]
        crush = post_vals.mean() - baseline_post
        ramps.append(ramp)
        crushes.append(crush)
        valid_events.append(ed)
    df = pd.DataFrame({
        "event_date": valid_events,
        "ramp": ramps,
        "crush": crushes,
    })
    # Compute summary statistics
    summary = {}
    for name, arr in {"ramp": np.array(ramps), "crush": np.array(crushes)}.items():
        if len(arr) > 0:
            mean_val = float(np.mean(arr))
            t_stat, t_p = stats.ttest_1samp(arr, popmean=0.0)
            # Wilcoxon signed rank test
            try:
                w_stat, w_p = stats.wilcoxon(arr)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            summary[f"mean_{name}"] = mean_val
            summary[f"t_stat_{name}"] = float(t_stat)
            summary[f"t_p_{name}"] = float(t_p)
            summary[f"w_stat_{name}"] = float(w_stat)
            summary[f"w_p_{name}"] = float(w_p)
        else:
            summary[f"mean_{name}"] = np.nan
            summary[f"t_stat_{name}"] = np.nan
            summary[f"t_p_{name}"] = np.nan
            summary[f"w_stat_{name}"] = np.nan
            summary[f"w_p_{name}"] = np.nan
    return df, summary


def main():
    # Paths
    vix_csv = os.path.join(os.getcwd(), "VIXCLS.csv")
    # Load VIX series
    vix_series = load_vix_data(vix_csv, start_date="2015-01-01")
    # Engineer features
    df = engineer_features(vix_series)
    # Define folds: use training up to year N and test on year N+1 for N in [2018, 2019, 2020, 2021, 2022]
    # Limit folds to keep runtime reasonable: use the most recent three years for validation
    years = [2019, 2020, 2021]
    fold_years = [(y, y + 1) for y in years if (y + 1) <= vix_series.index.year.max()]
    target_col = "iv"
    feature_cols = [c for c in df.columns if c != target_col]
    # Perform cross‑validation on the full feature set
    results = rolling_cross_validation(df, target_col, feature_cols, df.index, fold_years)
    res_df = pd.DataFrame([r.__dict__ for r in results])
    summary_table = res_df.groupby("model_name")[["rmse", "mae", "direction_hit"]].mean().sort_values("rmse")
    print("\nCross‑Validation Summary (all features, average over folds):")
    print(summary_table)
    # Perform ablations: drop the realised volatility proxy (iv_roll_std20) and drop lags
    ablation_results = []
    # Without realised volatility proxy
    feature_no_garch = [c for c in feature_cols if c != "iv_roll_std20"]
    results_no_garch = rolling_cross_validation(df, target_col, feature_no_garch, df.index, fold_years)
    ablation_results.extend(results_no_garch)
    # Without lagged IV features (only diff, roll_std20 and weekdays)
    feature_no_lags = [c for c in feature_cols if not c.startswith("iv_lag")]  # drop iv_lag1, iv_lag5, iv_lag10
    results_no_lags = rolling_cross_validation(df, target_col, feature_no_lags, df.index, fold_years)
    ablation_results.extend(results_no_lags)
    # Combine all results for ablations
    all_results_df = pd.DataFrame([r.__dict__ for r in results + results_no_garch + results_no_lags])
    # Compute summary tables
    summary_ablation = all_results_df.groupby("model_name")[["rmse", "mae", "direction_hit"]].mean().sort_values("rmse")
    print("\nCross‑Validation Summary including ablations (average over folds):")
    print(summary_ablation)
    # Event study
    event_dates = [pd.Timestamp(d) for d in ["2025-07-31", "2025-05-01", "2025-01-30", "2024-10-31"]]
    event_df, event_summary = perform_event_study(vix_series, event_dates)
    print("\nEvent Study Results:")
    print(event_df)
    print("\nEvent Study Summary:")
    for k, v in event_summary.items():
        print(f"{k}: {v:.4f}")
    # Save output tables to disk for later inspection
    res_df.to_csv("cv_fold_results.csv", index=False)
    summary_table.to_csv("cv_summary_table.csv")
    pd.DataFrame([r.__dict__ for r in results_no_garch]).to_csv("cv_fold_results_no_garch.csv", index=False)
    pd.DataFrame([r.__dict__ for r in results_no_lags]).to_csv("cv_fold_results_no_lags.csv", index=False)
    summary_ablation.to_csv("cv_summary_table_ablation.csv")
    event_df.to_csv("event_study_details.csv", index=False)
    with open("event_study_summary.txt", "w") as f:
        for k, v in event_summary.items():
            f.write(f"{k}: {v:.6f}\n")
    # Generate a static forecast plot for the last fold using ARIMAX without dynamic update
    if results:
        last_fold = fold_years[-1]
        train_end_year, test_year = last_fold
        train_mask = df.index.year <= train_end_year
        test_mask = df.index.year == test_year
        y_train = df.loc[train_mask, target_col]
        y_test = df.loc[test_mask, target_col]
        X_train = df.loc[train_mask, feature_cols]
        X_test = df.loc[test_mask, feature_cols]
        arimax_fit = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        # Static prediction: forecast for test period using full model and exogenous features
        pred_arimax_arr = arimax_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test)
        pred_arimax = pd.Series(pred_arimax_arr.values, index=y_test.index)
        plt.figure(figsize=(10, 4))
        plt.plot(y_train.index, y_train, label="Train", alpha=0.6)
        plt.plot(y_test.index, y_test, label="Test", color="black", linewidth=1.2)
        plt.plot(pred_arimax.index, pred_arimax, label="ARIMAX Forecast", color="red", linewidth=1.2)
        plt.title(f"ARIMAX Forecast vs Actual (Fold: train<= {train_end_year}, test= {test_year})")
        plt.ylabel("VIX")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.savefig("forecast_plot.png")
    print("\nAnalysis complete. Results saved to CSV files and plot saved to forecast_plot.png")


if __name__ == "__main__":
    main()
