# Intraday Strategies

One place to build, backtest and (later) run intraday strategies on NSE stocks.
Each strategy is its own folder and gets its own tab in the portal. The first one is **ADX intraday**.

> This app never places orders. It downloads public candle data, backtests rules, and shows results.

## What's here

| Folder / file | What it is |
|---|---|
| `ADX intraday strategy/` | The ADX strategy: `strategy.py` (rules), `config.yaml` (settings), `rules.html` (rules in plain words) |
| `common/` | Shared by every strategy: data download, indicators, trade simulation and costs, statistics, database |
| `portal/` | The web portal |
| `cli.py` | Command line: download data, run a backtest, start the portal |
| `data/nifty500.csv` | The stock list (Nifty 500, from NSE) |
| `tests/` | Automated checks |
| `var/` | Created when you run it: candles, database, logs. Never uploaded to GitHub |

## Run it on this PC

Double-click `run.bat`. The first time it installs everything (a few minutes), then opens http://localhost:8100.

1. **Data tab → Download.** Three years of 5-minute candles for 500 stocks. Takes about 5 hours the first time because Upstox limits the speed; later top-ups take about 10 minutes. You can stop and restart it.
2. **ADX intraday → Backtest → Run backtest.** Takes a few minutes.
3. Look at **Overview** (verdict and headline numbers), **Backtest** (running total and breakdowns), **Trades** (click any trade to see it on the chart) and **Stocks**.
4. **Settings** changes the rules or the costs for the next run. Old runs are kept so you can compare.

The same things from a terminal:

```bash
.venv\Scripts\python.exe cli.py download
```
```bash
.venv\Scripts\python.exe cli.py backtest adx
```
```bash
.venv\Scripts\python.exe -m pytest
```

## Running on the VPS

Same pattern as the Ignition & Coil scanner: pushing to GitHub builds a Docker image; the VPS only pulls it.

1. Push to GitHub. Wait for the **Docker image** action to go green (repo → Actions).
2. The first time only: on GitHub open your profile → Packages → `intraday-strategies` → Package settings → change visibility to **Public** (otherwise the VPS cannot pull it).
3. You need a web address for the portal that is different from the scanner's. Either a subdomain you own (add a DNS **A** record pointing to the VPS IP), or the free trick `strategies.<VPS IP with dashes>.sslip.io` (e.g. `strategies.203-0-113-7.sslip.io`).
4. In Hostinger → VPS → Docker Manager → **Compose from URL**, paste:
   `https://raw.githubusercontent.com/sudeep-sasikumar/Intraday-Strategies/master/docker-compose.yaml`
   and set two environment variables: `PORTAL_PASSWORD` (long) and `DOMAIN` (the address from step 3).
5. Open `https://<DOMAIN>`, sign in, then Data → Download and Backtest → Run backtest, as on the PC.

To update later: push to GitHub, wait for the action, press **Redeploy** in Docker Manager. Data and past runs survive (they live in a Docker volume).

## Paper trading (live signals, no orders)

1. On a strategy's **Settings** tab, load the values you want (for example the **Improved C** preset) and run a backtest to check them.
2. On its **Signals** tab press **Start paper trading**. The scanner then works by itself during market hours (09:15-15:30, Monday-Friday):
   - before the first scan of the day it tops up the candle history (about 10 minutes);
   - every 15 minutes it checks all 500 stocks for new signals; every 5 minutes it checks the open paper trades for exits;
   - each signal and each exit is recorded on the Signals tab and, if Telegram is set up, sent to your phone.
3. The Signals tab shows, for every paper trade, the live price when the alert was raised next to the entry price the backtest assumes. The difference is your real-world slippage.

**Skip stocks that have futures (F&O).** A setting under "Market filter and limits". The list of F&O stocks and its history come from NSE's daily futures files (`data/fno_membership.csv`). The scanner refreshes the list each morning from NSE; if NSE cannot be reached it carries on with the saved list.

It never places an order. It needs 15-minute (or longer) candles, because Upstox limits how often 500 stocks can be checked. The portal must be running for it to work, so use the VPS for real paper trading; on this PC it only runs while `run.bat` is open.

`python research/replay_check.py <run id> 12` replays past days through the scanner and compares its paper trades with a stored backtest run.

## How a backtest fills trades (on the cautious side)

- A setup is only known when its candle closes. The entry must trigger in the very next candle (setting: *Entry valid for*).
- 5-minute candles decide what happened first inside a 15-minute candle. If entry and stop are both touched in one 5-minute candle, it counts as a loss.
- A candle that opens beyond the entry price or the stop is filled at that open, not at the level.
- Strategy exits are filled at the open of the next 5-minute candle. Everything still open at 15:15 is closed.
- Costs on every trade: slippage on both fills, brokerage, STT, exchange and SEBI charges, stamp duty and GST.
- Every setup is taken with a fixed amount (default ₹1,00,000), on every stock, with no limit on how many are open at once. *Most trades open at once* tells you how much capital that would need.

What it cannot show: today's Nifty 500 list is used for the whole period (stocks that dropped out are missing), and thinly traded stocks fill worse in real life than here.

## Adding another strategy

1. Make a new folder next to `ADX intraday strategy/` (any name).
2. Put three files in it, copying the ADX ones as a template:
   - `config.yaml`: a unique `id`, a `name`, a one-line `summary`, `timeframe_min`, and its settings.
   - `strategy.py`: a `signals(bars, p)` function that returns, for every candle, whether there is a setup (+1 buy / −1 sell), the entry price, the stop price, and when to exit. Optionally `indicators(bars, p)` for lines under the trade chart.
   - `rules.html`: the rules in plain words for the Overview tab.
3. Restart the portal. The new strategy appears as a new master tab, with the same sub-tabs, and in **Compare**.

Nothing in `common/` or `portal/` needs to change.
