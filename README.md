## Crypto Modeling

This project extends an LSTM price forecaster with a simple question:
Do days with more tweets mentioning a cryptocurrency correspond to different market performance on those days (or the days after)?

Below is the plan, data flow, and reproducible steps to fetch tweet counts, align them with daily OHLCV data, and quantify the relationship.

Objectives

Collect daily tweet volume for a coin (e.g., “bitcoin” OR “$BTC” OR “#bitcoin”) over a given date range.

Join those counts to the existing price dataset by date.

Explore correlations (same-day and lagged), plus quick significance checks.

(Optional) feed tweet volume into the LSTM as an extra feature and measure lift.

Data Sources

Prices: CSV like Bitcoin Historical Data.csv
Columns include: Date, Price, Open, High, Low, Vol., Change %

Tweets (two options):

Without API keys: snscrape
 to count tweets by day.

With API keys: X (Twitter) API, /2 recent & full-archive search endpoints.

I’ll default to snscrape for simplicity/reproducibility. If we later need exact official counts, we can switch to the X API.

Environment Setup
# (Inside your venv)
pip install pandas numpy matplotlib scikit-learn torch
pip install snscrape==0.7.0.dev0  # or latest working version for your OS


For X API (optional):

pip install requests python-dotenv
# then set env var in .env
# X_BEARER_TOKEN=YOUR_TOKEN
