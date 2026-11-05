# OHLC Forecast app

This Streamlit app downloads the official NSE equity directory once daily and downloads fresh Yahoo Finance price history every time **Refresh data & forecast** is selected. It predicts the next daily session or intraday bar. It retains the supplied approach: 13 OHLC/lag/moving-average features and four separate `XGBRegressor` models using the original hyperparameters.

## Run

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## AWS deployment

The included `Dockerfile` runs Streamlit on port `8501`. It can be deployed to a container service such as AWS App Runner, or run on a free-tier eligible EC2 instance. Configure the service or security group to accept inbound TCP traffic on port `8501` (or place it behind a reverse proxy/load balancer). No credentials are stored in this repository; the application downloads public NSE and Yahoo Finance data when forecasts are requested.

The searchable dropdown contains all stocks in the official NSE equity directory. Select up to ten stocks for a batch CSV, then choose one of those stocks for a detailed chart. A “Yahoo Finance ticker” is the provider-specific symbol (for example, NSE `RELIANCE` is queried as `RELIANCE.NS`); the app builds those symbols internally, so no custom-ticker field is necessary.

The chart control can show every OHLC series or one price at a time. Orange predictions start from the latest actual point so the next forecast connects continuously to the actual line. Choose XGBoost, CatBoost, AdaBoost, Gradient Boosting, Random Forest, or Extra Trees; choosing **All algorithms (compare)** displays a ranked deviation table for the five-bar walk-forward test and uses the lowest recent close-deviation model for the next forecast. Every displayed prediction is trained only on data that preceded its validation bar. Yahoo Finance can be delayed or revise data; confirm a live decision against your broker or the NSE quote page. Forecasts are informational only, not trading advice.

The selected-stock CSV chooses the best algorithm separately for each stock, based on the lowest mean close deviation in its last five walk-forward predictions. The output contains the forecast date, stock name and NSE symbol, forecast and latest OHLC values, five-day close deviation, selected algorithm, and a status column.

