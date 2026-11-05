"""Interactive next-bar OHLC forecasts using Yahoo Finance and XGBoost."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor


st.set_page_config(page_title="OHLC Forecast", page_icon="📈", layout="wide")

# The feature set and XGBoost settings deliberately match the supplied script.
FEATURES = [
    "Open", "High", "Low", "Close", "Close_lag1", "Close_lag2", "Close_lag3",
    "Open_lag1", "High_lag1", "Low_lag1", "Volume", "MA5", "MA10",
]
TARGETS = ["Open", "High", "Low", "Close"]
ALGORITHMS = ["XGBoost", "CatBoost", "AdaBoost", "Gradient Boosting", "Random Forest", "Extra Trees"]
INTERVALS = {
    "1 day (next trading day)": ("1d", "5y"),
    "1 hour (next hour)": ("1h", "730d"),
    "30 minutes (next bar)": ("30m", "60d"),
    "15 minutes (next bar)": ("15m", "60d"),
    "5 minutes (next bar)": ("5m", "60d"),
}

NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"


@st.cache_data(ttl=86_400, show_spinner=False)
def load_nse_equity_list() -> pd.DataFrame:
    """Get the official NSE equity universe once daily instead of hard-coding stocks."""
    listings = pd.read_csv(NSE_EQUITY_LIST_URL)
    listings.columns = [str(column).strip() for column in listings.columns]
    if not {"SYMBOL", "NAME OF COMPANY"}.issubset(listings.columns):
        raise ValueError("The NSE equity-list file changed format.")
    listings = listings.dropna(subset=["SYMBOL", "NAME OF COMPANY"]).drop_duplicates("SYMBOL")
    listings["Yahoo ticker"] = listings["SYMBOL"].astype(str).str.strip() + ".NS"
    listings["Label"] = listings["NAME OF COMPANY"].astype(str).str.strip() + " — " + listings["SYMBOL"].astype(str).str.strip()
    return listings.sort_values("Label")[["Yahoo ticker", "Label"]]


def create_model(algorithm: str):
    """Build one regressor. XGBoost exactly preserves the supplied settings."""
    if algorithm == "XGBoost":
        return XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, random_state=42, objective="reg:squarederror", n_jobs=-1,
        )
    if algorithm == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=300, learning_rate=0.05, depth=5, random_seed=42,
                                 loss_function="RMSE", verbose=False, thread_count=-1)
    if algorithm == "AdaBoost":
        return AdaBoostRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    if algorithm == "Gradient Boosting":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    if algorithm == "Random Forest":
        return RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
    if algorithm == "Extra Trees":
        return ExtraTreesRegressor(n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def load_prices(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Download fresh Yahoo Finance OHLCV data for every forecast run."""
    frame = yf.download(ticker, interval=interval, period=period, auto_adjust=False,
                        progress=False, multi_level_index=False)
    if frame.empty:
        raise ValueError("Yahoo Finance returned no data. Check the ticker and interval.")
    frame = frame.rename_axis("Date").reset_index()
    return frame[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Close_lag1"] = df["Close"].shift(1)
    df["Close_lag2"] = df["Close"].shift(2)
    df["Close_lag3"] = df["Close"].shift(3)
    df["Open_lag1"] = df["Open"].shift(1)
    df["High_lag1"] = df["High"].shift(1)
    df["Low_lag1"] = df["Low"].shift(1)
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    return df[~df.index.duplicated(keep="last")]


def forecast(feature_df: pd.DataFrame, algorithm: str) -> tuple[dict[str, float], pd.DataFrame]:
    df = feature_df.dropna(subset=FEATURES).copy()
    if len(df) < 30:
        raise ValueError("At least 30 complete bars are needed for this forecast.")
    last_row = df.iloc[[-1]]
    for target in TARGETS:
        df[f"Target_{target}"] = df[target].shift(-1)
    train = df.dropna(subset=[f"Target_{t}" for t in TARGETS])
    predictions = {}
    for target in TARGETS:
        model = create_model(algorithm)
        model.fit(train[FEATURES], train[f"Target_{target}"])
        predictions[target] = float(model.predict(last_row[FEATURES])[0])
    # Make each forecast internally valid OHLC after four independent predictions.
    predictions["High"] = max(predictions["High"], predictions["Open"], predictions["Close"])
    predictions["Low"] = min(predictions["Low"], predictions["Open"], predictions["Close"])
    return predictions, df


def backtest_last_five(feature_df: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    """Walk-forward predictions: every row uses only earlier observations."""
    df = feature_df.dropna(subset=FEATURES).copy()
    rows = []
    # Forecast the known next bar after each of the five latest eligible origins.
    for origin in range(max(30, len(df) - 6), len(df) - 1):
        subset = df.iloc[: origin + 1].copy()
        for target in TARGETS:
            subset[f"Target_{target}"] = subset[target].shift(-1)
        train = subset.dropna(subset=[f"Target_{t}" for t in TARGETS])
        predicted = {}
        for target in TARGETS:
            model = create_model(algorithm)
            model.fit(train[FEATURES], train[f"Target_{target}"])
            predicted[target] = float(model.predict(subset.iloc[[-1]][FEATURES])[0])
        actual = df.iloc[origin + 1]
        rows.append({"Date": actual.name, **{f"Predicted {t}": predicted[t] for t in TARGETS},
                     **{f"Actual {t}": float(actual[t]) for t in TARGETS}})
    results = pd.DataFrame(rows)
    if not results.empty:
        results["Close deviation %"] = (results["Predicted Close"] - results["Actual Close"]).abs() / results["Actual Close"] * 100
    return results


def compare_algorithms(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Apply the identical five-bar walk-forward test to every selectable model."""
    summaries, validations = [], {}
    for algorithm in ALGORITHMS:
        try:
            validation = backtest_last_five(feature_df, algorithm)
            if validation.empty:
                raise ValueError("Insufficient eligible history")
            row = {"Algorithm": algorithm}
            for target in TARGETS:
                actual = validation[f"Actual {target}"].replace(0, np.nan)
                row[f"{target} deviation %"] = (
                    (validation[f"Predicted {target}"] - validation[f"Actual {target}"]).abs() / actual * 100
                ).mean()
            row["Average OHLC deviation %"] = np.mean([row[f"{target} deviation %"] for target in TARGETS])
            validations[algorithm] = validation
            summaries.append(row)
        except Exception as exc:
            summaries.append({"Algorithm": algorithm, "Status": f"Not run: {exc}"})
    ranking = pd.DataFrame(summaries)
    if "Close deviation %" in ranking:
        ranking = ranking.sort_values("Close deviation %", na_position="last").reset_index(drop=True)
    return ranking, validations


def make_forecast_record(ticker: str, stock_name: str, algorithm_choice: str = "All algorithms (compare)") -> dict[str, object]:
    """Produce the CSV-ready next-day forecast for a single NSE stock."""
    raw = load_prices(ticker, "1d", "5y")
    feature_df = prepare_features(raw)
    if algorithm_choice == "All algorithms (compare)":
        rankings, validations = compare_algorithms(feature_df)
        usable = rankings.dropna(subset=["Close deviation %"])
        if usable.empty:
            raise ValueError("No algorithm completed the five-day comparison")
        algorithm = str(usable.iloc[0]["Algorithm"])
        validation = validations[algorithm]
    else:
        algorithm = algorithm_choice
        validation = backtest_last_five(feature_df, algorithm)
    predicted, model_df = forecast(feature_df, algorithm)
    last = model_df.iloc[-1]
    forecast_date = next_label(model_df.index[-1], "1d")
    close_deviation = validation["Close deviation %"].mean() if not validation.empty else np.nan
    ohlc_deviations = []
    for target in TARGETS:
        actual = validation[f"Actual {target}"].replace(0, np.nan)
        ohlc_deviations.append(((validation[f"Predicted {target}"] - validation[f"Actual {target}"]).abs() / actual * 100).mean())
    return {
        "Forecast Date": forecast_date.strftime("%Y-%m-%d"),
        "Stock Name": stock_name,
        "NSE Symbol": ticker.removesuffix(".NS"),
        "Forecast Open": predicted["Open"],
        "Today Open": last["Open"],
        "Forecast High": predicted["High"],
        "Today High": last["High"],
        "Forecast Low": predicted["Low"],
        "Today Low": last["Low"],
        "Forecast Close": predicted["Close"],
        "Today Close": last["Close"],
        "Last 5 Days Close Deviation %": close_deviation,
        "Last 5 Days OHLC Average Deviation %": np.nanmean(ohlc_deviations),
        "Algorithm Used": algorithm,
        "Status": "Forecast created",
    }


def run_batch(listings: pd.DataFrame, progress_label: str) -> pd.DataFrame:
    """Keep going after individual ticker failures so one CSV covers the entire request."""
    rows: list[dict[str, object]] = []
    progress = st.progress(0, text=f"Preparing {progress_label}…")
    status = st.empty()
    total = len(listings)
    for position, (_, row) in enumerate(listings.iterrows(), start=1):
        ticker, label = row["Yahoo ticker"], row["Label"]
        status.caption(f"{progress_label}: {position:,} of {total:,} — {label}")
        try:
            rows.append(make_forecast_record(ticker, label))
        except Exception as exc:
            rows.append({
                "Forecast Date": "", "Stock Name": label, "NSE Symbol": ticker.removesuffix(".NS"),
                "Forecast Open": np.nan, "Today Open": np.nan, "Forecast High": np.nan, "Today High": np.nan,
                "Forecast Low": np.nan, "Today Low": np.nan, "Forecast Close": np.nan, "Today Close": np.nan,
                "Last 5 Days Close Deviation %": np.nan, "Last 5 Days OHLC Average Deviation %": np.nan,
                "Algorithm Used": "", "Status": f"Not forecast: {exc}",
            })
        progress.progress(position / total, text=f"{progress_label}: {position:,} of {total:,}")
    status.empty()
    progress.empty()
    return pd.DataFrame(rows)


def next_label(last_date: pd.Timestamp, interval: str) -> pd.Timestamp:
    if interval == "1d":
        candidate = last_date + pd.offsets.BDay(1)
        return pd.Timestamp(candidate)
    minutes = {"1h": 60, "30m": 30, "15m": 15, "5m": 5}[interval]
    return last_date + timedelta(minutes=minutes)


st.title("Next-bar OHLC forecast")
st.caption("Official NSE stock directory + fresh Yahoo Finance OHLC history + the supplied four-model XGBoost next-row method. Forecasts are estimates, not investment advice.")

left, right = st.columns([2, 1])
try:
    listings = load_nse_equity_list()
    ticker_labels = dict(zip(listings["Yahoo ticker"], listings["Label"]))
except Exception as exc:
    st.error(f"Could not load the official NSE stock directory: {exc}")
    st.stop()

with left:
    selected_tickers = st.multiselect("Select up to 10 NSE stocks (type to search)", list(ticker_labels),
                                      format_func=lambda symbol: ticker_labels[symbol], max_selections=10,
                                      placeholder="Search by company or NSE symbol")
    chart_ticker = st.selectbox("Stock to show in the detailed chart", selected_tickers,
                                format_func=lambda symbol: ticker_labels[symbol], index=None,
                                disabled=not selected_tickers,
                                placeholder="Select stock above")
with right:
    display_interval = st.selectbox("Forecast interval", list(INTERVALS))
    chart_price = st.selectbox("Chart price", ["All OHLC", "Open", "High", "Low", "Close"])
    selected_algorithm = st.selectbox("Forecast algorithm", ["All algorithms (compare)"] + ALGORITHMS,
                                      help="All algorithms runs a five-bar walk-forward test for every model and selects the best recent close result for the chart.")
    run = st.button("Forecast selected chart stock", type="primary", use_container_width=True, disabled=chart_ticker is None)
    selected_batch_run = st.button("Download selected stocks CSV (max 10)", use_container_width=True,
                                   disabled=not selected_tickers,
                                   help="Forecasts every selected stock and downloads the requested CSV structure.")

if selected_batch_run:
    selected_listings = listings[listings["Yahoo ticker"].isin(selected_tickers)]
    selected_results = run_batch(selected_listings, "Selected-stock forecast")
    successful = (selected_results["Status"] == "Forecast created").sum()
    st.success(f"Selected-stock batch complete: {successful:,} forecasts created out of {len(selected_results):,} stocks.")
    st.download_button(
        "Download selected stocks forecast CSV",
        data=selected_results.to_csv(index=False).encode("utf-8"),
        file_name="selected_nse_next_day_ohlc_forecasts.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.dataframe(selected_results, use_container_width=True, hide_index=True)

if run:
    ticker = chart_ticker
    interval, period = INTERVALS[display_interval]
    try:
        with st.spinner("Downloading latest history and training the selected forecast model(s)..."):
            raw = load_prices(ticker, interval, period)
            feature_df = prepare_features(raw)
            if selected_algorithm == "All algorithms (compare)":
                rankings, validations = compare_algorithms(feature_df)
                usable = rankings.dropna(subset=["Close deviation %"])
                if usable.empty:
                    raise ValueError("None of the algorithms could complete the five-bar comparison.")
                active_algorithm = usable.iloc[0]["Algorithm"]
                validation = validations[active_algorithm]
            else:
                active_algorithm = selected_algorithm
                validation = backtest_last_five(feature_df, active_algorithm)
                rankings = None
            predicted, model_df = forecast(feature_df, active_algorithm)
        last = model_df.iloc[-1]
        forecast_date = next_label(model_df.index[-1], interval)
        st.success(f"Updated from Yahoo Finance: {ticker}. Last completed bar: {model_df.index[-1]:%d %b %Y %H:%M}. Forecast model: {active_algorithm}.")
        st.caption("Stock names: official NSE Equity List (refreshed daily). OHLCV: Yahoo Finance, downloaded now. The timestamp above is the newest bar returned by the provider; verify live trading values with NSE or your broker before acting.")
        if rankings is not None:
            st.subheader("Algorithm comparison — last five walk-forward forecasts")
            st.caption("Lower deviation is better. The chart and next forecast use the algorithm with the lowest recent close deviation.")
            numeric_columns = [column for column in rankings.columns if column.endswith("%")]
            st.dataframe(rankings.style.format({column: "{:.2f}%" for column in numeric_columns}), use_container_width=True, hide_index=True)
        cols = st.columns(4)
        for col, key in zip(cols, TARGETS):
            col.metric(f"Forecast {key}", f"{predicted[key]:,.2f}")
        change = (predicted["Close"] - last["Close"]) / last["Close"] * 100
        st.metric("Expected close change", f"{change:+.2f}%", f"Last close: {last['Close']:,.2f}")

        chart_data = model_df.tail(5).copy()
        fig = go.Figure()
        selected_prices = TARGETS if chart_price == "All OHLC" else [chart_price]
        palette = {"Open": "#1f77b4", "High": "#9467bd", "Low": "#d62728", "Close": "#16803a"}
        for price in selected_prices:
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data[price], mode="lines+markers",
                                     name=f"Actual {price}", line=dict(color=palette[price], width=3)))
        if not validation.empty:
            for price in selected_prices:
                fig.add_trace(go.Scatter(x=validation["Date"], y=validation[f"Predicted {price}"], mode="lines+markers",
                                         name=f"Predicted {price}", line=dict(color="#f58220", dash="dot")))
        # Starts at the latest actual value: orange forecast visibly joins the actual line.
        for price in selected_prices:
            fig.add_trace(go.Scatter(x=[model_df.index[-1], forecast_date], y=[last[price], predicted[price]], mode="lines+markers+text",
                                     name=f"Next {price} forecast", text=["", f"{price} forecast"], textposition="top center",
                                     line=dict(color="#f58220", width=3), marker=dict(color="#f58220", size=10)))
        fig.update_layout(title="Last five bars: actual (colours) vs predicted (orange), plus connected next forecast", xaxis_title="Time", yaxis_title="Price", hovermode="x unified", height=520)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader(f"Last five walk-forward validations — {active_algorithm}")
        if validation.empty:
            st.info("There is not enough eligible history to calculate five validations.")
        else:
            st.dataframe(validation.style.format({c: "{:.2f}" for c in validation.columns if c != "Date"}), use_container_width=True, hide_index=True)
            st.caption(f"Mean absolute close deviation: {validation['Close deviation %'].mean():.2f}%  •  Maximum: {validation['Close deviation %'].max():.2f}%")
    except Exception as exc:
        st.error(str(exc))
else:
    st.info("Search for an NSE company or symbol, choose the interval and chart price, then select **Refresh data & forecast**.")

