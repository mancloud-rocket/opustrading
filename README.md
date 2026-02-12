# OPUS Trading Bot v3.0

Paper trading bot for Polymarket's 15-minute BTC/USD binary markets.

## Strategy

**Cheap Token + BTC Momentum** — Buys UP/DOWN tokens when BTC shows directional momentum from Binance and the corresponding token is still cheap on Polymarket.

| Parameter | Value |
|-----------|-------|
| BTC Threshold | >= 0.065% from market start |
| Token Price | $0.30 - $0.60 |
| Entry Window | Minute 5-10 |
| Take Profit | Token >= $0.97 |
| Stop Loss | None (hold to resolution) |
| Bet Size | $20/trade |

**Validated on live data (Feb 11, 2026):** 13 trades, 69% WR, +$56.71 PnL, $29 max drawdown.

## Project Structure

```
opustrading/
├── python/
│   ├── live/                    # Live bot modules
│   │   ├── settings.py          # All tunable parameters
│   │   ├── trader.py            # Trading engine + paper execution
│   │   ├── signal_processor.py  # Real-time BTC signal analysis
│   │   ├── binance_feed.py      # Binance BTC price feed
│   │   ├── polymarket_client.py # Polymarket market discovery + prices
│   │   ├── dashboard.py         # Real-time stats dashboard (CSV-backed)
│   │   └── health.py            # System health monitor
│   ├── backtest/                # Backtesting framework
│   │   ├── backtester.py        # Backtest engine
│   │   └── lead_lag.py          # Lead-lag analysis
│   ├── data/                    # Data loading + Binance client
│   │   ├── parser.py            # Polymarket CSV parser
│   │   └── binance_client.py    # Binance historical data
│   ├── features/
│   │   └── engine.py            # Feature engineering
│   ├── strategy/
│   │   └── strategies.py        # Strategy definitions
│   ├── config.py                # Global config
│   ├── run_live.py              # Live bot entry point
│   ├── run_backtest.py          # Backtest entry point
│   └── requirements.txt
├── data/
│   ├── live_trades.csv          # Paper trade log (generated)
│   └── live_log.csv             # Tick-by-tick log (generated)
└── .gitignore
```

## Setup

```bash
cd python
pip install -r requirements.txt
```

## Usage

### Live Paper Trading

```bash
cd python
python run_live.py
```

The bot will:
1. Connect to Binance for real-time BTC prices
2. Discover active 15-min BTC markets on Polymarket
3. Monitor BTC momentum and token prices
4. Log paper trades to `data/live_trades.csv`
5. Display a real-time dashboard every 5 minutes with session stats

### Backtesting

```bash
cd python
python run_backtest.py
```

Runs historical backtests using `data/prices.csv` (Polymarket tick data) and Binance klines.

## Configuration

All parameters are in `python/live/settings.py`:

```python
BTC_THRESHOLD = 0.00065     # 0.065% min BTC return
TAKE_PROFIT_PRICE = 0.97    # TP at $0.97
ENTRY_PRICE_MIN = 0.30      # Min token price to buy
ENTRY_PRICE_MAX = 0.60      # Max token price to buy
ENTRY_SECOND_MIN = 300      # Earliest entry (min 5)
ENTRY_SECOND_MAX = 600      # Latest entry (min 10)
BET_SIZE = 20.00             # $ per trade
MAX_DAILY_LOSS = 100.00      # Daily loss limit
```

## Dashboard

The dashboard auto-renders every 5 minutes and shows:
- Session stats (WR, PnL, profit factor, max drawdown)
- All-time cumulative stats
- Performance by exit type (TAKE_PROFIT vs RESOLUTION)
- Hourly performance breakdown (ET timezone)
- Last 10 trades with details
- Live BTC threshold sweep analysis

## How It Works

1. **Market Discovery**: Finds active 15-min BTC UP/DOWN markets on Polymarket via Gamma API
2. **BTC Reference**: Fetches exact BTC open price at market start from Binance 1-min klines
3. **Signal**: Computes `btc_return = (btc_now - btc_start) / btc_start` every second
4. **Entry**: If `|btc_return| >= 0.065%` and token price is in $0.30-$0.60 range during minutes 5-10, buys the momentum side
5. **Exit**: Takes profit at $0.97 or holds to market resolution (15 min)
6. **Logging**: Every tick logged to CSV; trades logged with full metadata

## Data Sources

- **Binance**: REST API for BTC/USDT spot price (no API key required)
- **Polymarket**: Gamma API for market discovery, CLOB API for token prices
