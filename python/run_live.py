"""
OPUS Trading Bot v3.0 - Live Paper Trading

Estrategia: Cheap Token + BTC Reversal Stop
  - Monitorea mercados BTC 15-min en Polymarket
  - Obtiene precio BTC en real-time desde Binance
  - Entry: BTC ret > 0.08% + token en [0.25, 0.50] + minuto 3-7
  - Exit: BTC reversal (-0.10%) o resolucion
  - Logs de paper trading en data/live_trades.csv

Uso:
  cd python
  pip install -r requirements.txt
  python run_live.py
"""

import sys
import time
import signal
from datetime import datetime, timezone
from pathlib import Path

# Agregar directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

from live.binance_feed import BinanceFeed
from live.polymarket_client import PolymarketClient, MarketInfo
from live.trader import Trader, STRATEGY_CONFIG
from live.dashboard import Dashboard
from live.health import HealthMonitor
import live.settings as settings


# ============================================================================
# Configuracion (desde settings.py)
# ============================================================================

POLL_INTERVAL = settings.POLL_INTERVAL
MARKET_CHECK_INTERVAL = settings.MARKET_CHECK_INTERVAL
STATUS_INTERVAL = settings.STATUS_INTERVAL
VERBOSE = settings.VERBOSE


# ============================================================================
# Colores para consola
# ============================================================================

class C:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


def log(level: str, msg: str):
    """Log con timestamp y color."""
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    colors = {
        "INFO": C.CYAN,
        "TRADE": C.GREEN,
        "EXIT": C.YELLOW,
        "WARN": C.YELLOW,
        "ERROR": C.RED,
        "PRICE": C.GRAY,
        "SIGNAL": C.BOLD,
    }
    color = colors.get(level, C.RESET)
    print(f"{color}[{now}] [{level:6s}] {msg}{C.RESET}")


# ============================================================================
# Banner
# ============================================================================

def print_banner():
    cfg = STRATEGY_CONFIG
    ml_status = "ML PREDICTOR" if cfg.get("use_ml_predictor") else "THRESHOLDS"
    sl_pct = cfg.get("token_stop_loss_pct", 0)
    sl_str = f"{sl_pct*100:.0f}% token drop" if sl_pct else "DESACTIVADO"
    print(f"""
{C.GREEN}+======================================================================+
|  OPUS Trading Bot v3.3 - Paper Trading                              |
|  Estrategia: GradientBoosting + BTC Features (prediccion continua)  |
+======================================================================+
|                                                                      |
|  ENTRY:                                                              |
|    BTC threshold: >= {cfg['btc_threshold']*100:.3f}% (HARD filter, ML y fallback)|
|    ML confidence: >= {cfg.get('ml_min_confidence', 0.58)*100:.0f}% (si modelo disponible)              |
|    Token price: ${cfg['entry_price_min']:.2f} - ${cfg['entry_price_max']:.2f} (max ${cfg.get('entry_price_max_strong', 0.65):.2f} si BTC > 0.15%)   |
|    Ventana: minuto {cfg['entry_second_min']//60}-{cfg['entry_second_max']//60}                                            |
|                                                                      |
|  EXIT:                                                               |
|    Take profit: token >= ${cfg.get('take_profit_price', 0.97):.2f}                                |
|    Stop loss: {sl_str:<45}|
|    Resolution: hold hasta que el mercado resuelva                    |
|                                                                      |
|  RISK:                                                               |
|    Bet size: ${cfg['bet_size']:.2f} | Fee: {cfg['fee_rate']*100:.0f}% por lado                          |
|    Max daily loss: ${cfg['max_daily_loss']:.2f} | Max 1 trade/mercado                |
|                                                                      |
|  MODO: PAPER TRADING (solo logs, sin ejecucion real)                 |
+======================================================================+{C.RESET}
""")


# ============================================================================
# Main loop
# ============================================================================

def main():
    print_banner()

    # Inicializar componentes
    data_dir = str(Path(__file__).parent.parent / "data")
    binance = BinanceFeed()
    polymarket = PolymarketClient()
    trader = Trader(data_dir=data_dir)
    dashboard = Dashboard(trader, data_dir=data_dir)
    health = HealthMonitor()

    # Shutdown handler
    running = [True]

    def on_shutdown(signum, frame):
        running[0] = False
        log("INFO", "Shutdown solicitado...")

    signal.signal(signal.SIGINT, on_shutdown)

    # Primera conexion
    log("INFO", "Conectando a Binance...")
    btc = binance.fetch_price()
    if btc:
        log("INFO", f"BTC/USDT: ${btc:,.2f}")
        health.record("binance", True)
    else:
        log("WARN", "No se pudo obtener precio BTC. Reintentando...")
        health.record("binance", False, "initial connection failed")

    log("INFO", "Buscando mercado activo en Polymarket...")
    market = polymarket.find_current_market()

    if market:
        log("INFO", f"Mercado: {market.title}")
        log("INFO", f"  Slug: {market.slug}")
        log("INFO", f"  Elapsed: {market.elapsed_seconds:.0f}s "
            f"({market.market_minute:.1f} min)")
        log("INFO", f"  UP token: {market.up_token_id[:20]}...")
        log("INFO", f"  DOWN token: {market.down_token_id[:20]}...")
        health.record("polymarket_discovery", True)

        # Obtener BTC price al inicio EXACTO del mercado (via kline)
        btc_at_start = binance.fetch_price_at_market_start(market.start_time)
        if btc_at_start:
            log("INFO", f"  BTC al inicio del mercado (kline): ${btc_at_start:,.2f}")
            log("INFO", f"  BTC actual: ${btc:,.2f} "
                f"(diff: {((btc - btc_at_start) / btc_at_start * 100):+.4f}%)")
        else:
            btc_at_start = btc
            log("WARN", f"  No se pudo obtener BTC historico, usando actual: ${btc:,.2f}")

        trader.on_new_market(market, btc_at_start)
    else:
        secs = polymarket.seconds_until_next_market()
        log("WARN", f"No se encontro mercado activo. "
            f"Proximo en {secs:.0f}s")
        health.record("polymarket_discovery", False, "no market found")

    # State
    last_market_check = time.time()
    last_status_print = 0.0
    current_market_slug = market.slug if market else None

    log("INFO", "Iniciando loop principal...")
    print()

    # ================================================================
    # MAIN LOOP
    # ================================================================

    while running[0]:
        try:
            now = time.time()

            # --------------------------------------------------------
            # 1. Market discovery / transition
            # --------------------------------------------------------
            if now - last_market_check >= MARKET_CHECK_INTERVAL:
                last_market_check = now

                new_slug = polymarket.generate_current_slug()

                if new_slug != current_market_slug:
                    # Market transition
                    if market and trader.position:
                        # Cerrar posicion del mercado anterior
                        prices = polymarket.fetch_prices()
                        btc = binance.get_price_or_cached()
                        trader.on_market_close(market, prices, btc)
                        log("EXIT", f"Mercado cerrado: {market.title}")

                    # Buscar nuevo mercado
                    old_market = market
                    market = polymarket.find_current_market()

                    if market:
                        current_market_slug = market.slug
                        btc = binance.get_price_or_cached()
                        # BTC al inicio exacto del mercado
                        btc_start = binance.fetch_price_at_market_start(
                            market.start_time
                        )
                        if btc_start:
                            log("INFO", f"  BTC inicio mercado: ${btc_start:,.2f} "
                                f"(actual: ${btc:,.2f})")
                        else:
                            btc_start = btc
                        trader.on_new_market(market, btc_start)

                        # Dashboard header para nuevo mercado
                        print(dashboard.render_header(market, btc))
                    else:
                        current_market_slug = new_slug
                        secs = polymarket.seconds_until_next_market()
                        if secs > 30:
                            log("INFO", f"Esperando proximo mercado "
                                f"({secs:.0f}s)...")

                elif market is None:
                    # Reintentar
                    market = polymarket.find_current_market()
                    if market:
                        current_market_slug = market.slug
                        btc = binance.get_price_or_cached()
                        btc_start = binance.fetch_price_at_market_start(
                            market.start_time
                        )
                        if not btc_start:
                            btc_start = btc
                        trader.on_new_market(market, btc_start)
                        log("INFO", f"Mercado encontrado: {market.title}")

            # --------------------------------------------------------
            # 2. Si no hay mercado, esperar
            # --------------------------------------------------------
            if market is None or market.is_expired:
                time.sleep(POLL_INTERVAL)
                continue

            # --------------------------------------------------------
            # 3. Fetch data
            # --------------------------------------------------------
            btc = binance.get_price_or_cached()
            health.record("binance", btc is not None,
                          "price fetch failed" if btc is None else "")

            prices = polymarket.fetch_prices()
            health.record("polymarket_prices", prices.is_valid,
                          prices.error or "" if not prices.is_valid else "")

            # Market closed?
            if prices.market_closed:
                log("INFO", f"Mercado cerrado por API")
                if trader.position:
                    trader.on_market_close(market, prices, btc)
                market = None
                current_market_slug = None
                time.sleep(POLL_INTERVAL)
                continue

            # --------------------------------------------------------
            # 4. Process tick (signal engine)
            # --------------------------------------------------------
            event = trader.process_tick(market, prices, btc)

            # --------------------------------------------------------
            # 5. Print status
            # --------------------------------------------------------
            if "ENTRY" in event:
                pos = trader.position
                if pos:
                    btc_ret = trader._calc_btc_return(btc)
                    ret_str = f"{btc_ret*100:+.3f}%" if btc_ret else "?"
                    log("TRADE",
                        f"ENTRY {pos.side} @ {pos.entry_price:.2f} "
                        f"| min {pos.entry_minute} "
                        f"| BTC ret: {ret_str} "
                        f"| Market: {market.title}")

            elif "EXIT" in event:
                # Trade was just closed, show result
                if trader.trade_history:
                    last = trader.trade_history[-1]
                    pnl_color = C.GREEN if last.pnl_net > 0 else C.RED
                    log("EXIT",
                        f"{last.reason} | {last.side} "
                        f"entry={last.entry_price:.2f} "
                        f"exit={last.exit_price:.2f} "
                        f"| PnL: {pnl_color}${last.pnl_net:+.2f}{C.RESET} "
                        f"| Balance: ${trader.session_balance:+.2f}")

            elif now - last_status_print >= STATUS_INTERVAL:
                last_status_print = now
                status = trader.get_status_line(market, prices, btc)
                # Only print during active window or if position open
                elapsed = market.elapsed_seconds
                if trader.position or (120 <= elapsed <= 500):
                    log("PRICE", status)
                elif int(elapsed) % 60 < STATUS_INTERVAL + 1:
                    # Print once per minute otherwise
                    log("PRICE", status)

            # --------------------------------------------------------
            # 6. Periodic full stats + health
            # --------------------------------------------------------
            if dashboard.should_show_full_stats(interval=settings.FULL_STATS_INTERVAL):
                print(dashboard.render_full_stats())
                print("  HEALTH:")
                print(health.get_status_summary())
                print()

            # --------------------------------------------------------
            # 7. Daily loss limit
            # --------------------------------------------------------
            if trader.daily_stats.is_loss_limit_hit:
                log("WARN",
                    f"DAILY LOSS LIMIT alcanzado: "
                    f"${trader.daily_stats.pnl_net:.2f}. "
                    f"Pausando trades por hoy.")

            # --------------------------------------------------------
            # Sleep (con backoff si hay errores)
            # --------------------------------------------------------
            wait = max(POLL_INTERVAL, health.get_wait_recommendation())
            time.sleep(wait)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log("ERROR", f"Error en loop: {e}")
            time.sleep(2)

    # ================================================================
    # SHUTDOWN
    # ================================================================

    print()
    log("INFO", "Cerrando bot...")

    if trader.position and market:
        prices = polymarket.fetch_prices()
        btc = binance.get_price_or_cached()
        trader.on_market_close(market, prices, btc)
        log("EXIT", "Posicion cerrada por shutdown")

    trader.print_session_summary()

    print(f"""
{C.YELLOW}+======================================================================+
|  SHUTDOWN COMPLETO                                                   |
|  Balance sesion: ${trader.session_balance:+.2f}{' ' * (46 - len(f'${trader.session_balance:+.2f}'))}|
|  Trades: {trader.daily_stats.trades} ({trader.daily_stats.wins}W / {trader.daily_stats.losses}L){' ' * max(0, 52 - len(f'{trader.daily_stats.trades} ({trader.daily_stats.wins}W / {trader.daily_stats.losses}L)'))}|
|  Logs: data/live_trades.csv, data/live_log.csv                      |
+======================================================================+{C.RESET}
""")


if __name__ == "__main__":
    main()

