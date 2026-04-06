# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A SPY ETF data analysis toolkit with standalone Python scripts. No build system or test framework.

## Dependencies

```bash
pip install yfinance pandas numpy scipy scikit-learn requests python-dotenv pyarrow
```

## Scripts and Usage

### `spy_downloader.py` — Download historical data
```bash
# Default: yfinance, 1h interval, last 6 months → saves to ./spy_data/
python spy_downloader.py

# 15-minute bars, last 60 days
python spy_downloader.py --source yfinance --interval 15m --period 60d

# Date range
python spy_downloader.py --source yfinance --interval 1h --start 2024-01-01 --end 2024-12-31

# Alpha Vantage (requires API key)
python spy_downloader.py --source alpha_vantage --interval 30min --apikey KEY --months 6

# Polygon.io (requires API key)
python spy_downloader.py --source polygon --multiplier 15 --timespan minute --apikey KEY --months 3
```
Outputs: CSV + Parquet files named `SPY_{interval}_{source}_{timestamp}` in `./spy_data/`.

### `spy_analysis_1h.py` — Day-type predictor
```bash
python spy_analysis_1h.py spy_data/SPY_1h_yfinance_*.csv
python spy_analysis_1h.py datos.csv --output reporte.html --umbral-ret -0.3 --umbral-gap 0.15
```
Reads 1h CSV, engineers daily features, runs KMeans clustering to classify day types, computes next-day up/down probabilities with Wilson confidence intervals. Generates a self-contained HTML report with interactive charts.

### `spy_analysis_live.py` — Live day analyzer
```bash
python spy_analysis_live.py spy_data/SPY_1h_yfinance_cummulative.csv spy_data/SPY_15m_yfinance_cummulative.csv
python spy_analysis_live.py datos_1h.csv datos_15m.csv --output reporte.html
```
Takes both cumulative CSVs (including today's partial data) and generates an HTML report with: current day characterization, prediction for today's close, prediction for tomorrow's direction, and historical analogues. Run after `--update` on both intervals to have the latest data.

### `spy_analysis_maxmin.py` — Intraday high/low timing
```bash
python spy_analysis_maxmin.py spy_data/SPY_15m_yfinance_*.csv
python spy_analysis_maxmin.py datos_15m.csv --output reporte.html --ticker SPY --tz US/Eastern
```
Works with any intraday granularity (15m, 30m, 1h). Finds which time slots most frequently produce the daily high/low. Generates a self-contained HTML report.

## Architecture

**Data flow**: `spy_downloader.py` → `spy_data/*.csv` → analysis scripts → `*.html` reports.

**`spy_analysis_1h.py` pipeline:**
1. `load_and_build_daily()` — aggregates intrabar data to daily OHLCV + features (ret, range_pct, gap, close_pos, morning/afternoon return splits, vol_trend)
2. KMeans clustering on scaled features to identify day archetypes
3. Statistical analysis per cluster (Wilson CI on next-day direction)
4. HTML report generation with embedded JSON data for Plotly charts

**`spy_analysis_live.py` pipeline:**
1. `load_csv()` — loads both CSVs (1h and 15m), converts to NY timezone
2. `build_daily_history()` — builds historical daily features from 1h data, excluding today
3. `extract_today_features()` — computes current-day features from partial 1h + 15m data (uses 15m for finer close price and intraday trend)
4. `predict_today_close()` — statistical prediction of today's close based on historical analogues
5. HTML report generation

**`spy_analysis_maxmin.py` pipeline:**
1. `load_data()` — loads CSV, converts to NY timezone, validates columns
2. `build_daily_extremes()` — per-day records of which time slot had the high/low
3. `compute_freq_tables()` — frequency counts and percentages per time slot
4. HTML report generation with frequency bar charts and cumulative distribution

**Data format**: All scripts expect CSVs with a datetime index (UTC-aware) and columns `open, high, low, close, volume`. This is the format produced by `spy_downloader.py`.

## yfinance Historical Limits

- 1m → max 7 days; 15m/30m → max 60 days; 1h → max 730 days; 1d+ → full history
- Cumulative CSVs (`spy_data/*_cummulative.csv`) aggregate multiple downloads to extend history beyond these limits.
