# Project Chimera v4

## Project Overview
Project Chimera v4 is a high-frequency, automated quantitative trading system designed for Kalshi. It operates in both live and shadow (paper trading) modes, evaluating real-time probabilities across multiple asset classes (Sports, Weather, Crypto) and executing trades based on an expected value (EV) framework. The system architecture is built around an asynchronous dual-listener model to handle real-time market data concurrently with Kalshi order book updates.

### Key Components
- **Core Engine (`chimera/trading/autotrade.py` / `shadow.py`)**: An `asyncio`-driven dual-listener architecture that processes external data streams (Stream A) alongside Kalshi WS ticker updates (Stream B). It recalculates dynamic `p_true` values and triggers mass-cancels (Velocity Kill Switches) when edges disappear.
- **Oracles**: 
  - **NBA Live (`nba_live_v4.py`)**: Aggressive ESPN API poller factoring in possession and clock variables for live probability blending.
  - **NHL Live (`nhl_live_v4.py`)**: Monitors ESPN for extreme variance hockey states (e.g., pulled goalies, power plays) to dynamically adjust required trading edges.
  - **Crypto Spot (`crypto_oracle_v4.py`)**: High-frequency Binance WebSocket listener tracking flash crash threshold swings to safeguard resting orders.
  - **Weather (`weather_oracle_v4.py`)**: Cache-busting NOAA gridpoint poller calculating dynamic temperature volatility (`sigma`) over specified windows.
- **Execution & Fees (`chimera/fees.py`)**: Advanced EV calculators integrating adverse selection penalties against the conditional `p_true` for maker orders.

## Building and Running
The bot uses an `.env` file structure (specifically `env.list`) to manage Kalshi API credentials, data provider keys (ESPN, Odds API, Binance), and GCP configurations.

### Running the Shadow Trader
1. Ensure the environment is loaded:
   `export $(grep -v '^#' env.list | xargs)`
2. Run the shadow daemon:
   `python3 agents/kalshi-meta/orchestrator/run_shadow.py --config config/signal_registry.json --poll-seconds 1.5`

### Auditing
- Verify oracle functionality and RSA key configurations using:
  `python3 audit_v4.py`

### Going Live
To execute live trades, you must use the live orchestrator script with explicit confirmation and strict contract sizing:
`python3 agents/kalshi-meta/orchestrator/run_live_daemon.py --size-contracts 1 --force-size-contracts`

## Development Conventions
- **Asynchronous IO**: Use `asyncio` and `aiohttp` or `websockets` for all network requests in the trading path to prevent execution blocking.
- **Strict Anti-Spoofing**: Limit prices are mathematically anchored to `(dynamic_p_true * 100) - required_edge`. The bot never blindly sets prices at `yes_bid + 1`.
- **Kill Switches**: Any massive move in underlying oracles must instantly trigger mass-cancellations before updating Kalshi resting orders.
- **Safety First**: Live execution modes should always have a dry-run equivalent or strictly bounded `--size-contracts` arguments to prevent runaway losses.
