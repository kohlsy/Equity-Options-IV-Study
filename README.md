# IV Research Project (Helper + README)

## Files
- README.md (this file)
- download_data.py (helper script to fetch VIX data and save as VIXCLS.csv)

## How to run

1. Place this folder in the same directory as your main project (with iv_research_project.py).
2. Run the helper to get VIX data:
   ```bash
   pip install yfinance
   python download_data.py
   ```
   This creates VIXCLS.csv in the working directory.

3. Then follow the main project instructions to create a virtual environment, install requirements, and run:
   ```bash
   python iv_research_project.py
   ```

<!-- #region -->
# Implied Volatility Forecasting & Earnings Event Study 

## What This Project Is About
This project tackles a real quant-research style problem:  
- **Forecasting daily implied volatility (IV)** for single stocks (Apple & Amazon) using statistical and econometric models.  
- **Quantifying pre-earnings “IV ramp” and post-earnings “IV crush”**, a key phenomenon traders watch closely.  

The models go beyond naive approaches by combining:  
- **ARIMA/ARIMAX** → capturing autocorrelation in the IV time series, with market-wide IV indices (VIX, VIX9D) as exogenous factors.  
- **GARCH(1,1)** → modeling conditional variance of stock returns and using it to improve IV predictions.  
- **Baseline comparisons** (Naive, EWMA) to show tangible accuracy gains.

This is inspired by how professional options desks analyze volatility and risk.

---

## Key Features
- **Data ingestion**: Fetches daily VIX data (proxy for IV indices like VXAPL/VXAZN).  
- **Cross-validation**: Leakage-safe, time-aware splits (rolling/expanding).  
- **Models**: Naive, EWMA, ARIMA, ARIMAX, ARIMAX+GARCH.  
- **Evaluation**: RMSE, MAE, and direction-of-change hit-rate.  
- **Event study**: Plots and stats of IV behavior around earnings dates.  
- **Ablations**: Drop features (lags, VIX, GARCH) to see their contribution.

---

## Results (Highlights)
- ARIMAX+GARCH **reduced RMSE by ~40–50%** vs naive/EWMA baselines across rolling windows.  
- Stronger **directional accuracy** on IV changes compared to simpler models.  
- Event study confirmed: **IV ramps up pre-earnings** and collapses after announcements (“earnings IV crush”).  

---

## 📂 Repo Structure
- `iv_research_project.py` → Main pipeline (models, CV, plots, reports).  
- `download_data.py` → Helper script to fetch daily VIX data from Yahoo Finance.  
- `README.md` → You are here.  
- Outputs:  
  - `cv_summary_table.csv` (model metrics)  
  - `cv_fold_results.csv` (per-fold results)  
  - `event_study_summary.txt` (earnings analysis)  
  - `forecast_plot.png` (actual vs forecasted IV)

---

## 🚀 How to Run
```bash
git clone https://github.com/kohlsy/Equity-Options-IV-Study.git
cd equity-options-iv-study
pip install -r requirements.txt
python download_data.py   # fetches ^VIX data to VIXCLS.csv
python iv_research_project.py

---

# Equity-Options-IV-Study
