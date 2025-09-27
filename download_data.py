import yfinance as yf

# Fetch VIX data (^VIX) from Yahoo Finance
ticker = "^VIX"
data = yf.download(ticker, start="2010-01-01", end="2025-01-01")

# Save to CSV in the format expected by the main project
data[['Close']].to_csv("VIXCLS.csv")
print("Saved VIXCLS.csv with daily Close prices for ^VIX (Yahoo Finance).")
