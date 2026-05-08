"""Exchange factory and symbol normalisation helpers.

We use ``ccxt.async_support`` for REST and ``ccxt.pro`` (when installed) for
streaming. Symbols follow the ccxt unified ``BASE/QUOTE`` format; per-exchange
quirks (e.g. KuCoin spelling ``BTC-USDT``) are handled by ccxt itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ccxt.async_support as ccxt_async

try:
    import ccxt.pro as ccxt_pro

    _HAS_CCXT_PRO = True
except ImportError:
    ccxt_pro = None  # type: ignore[assignment]
    _HAS_CCXT_PRO = False


# Curated registry of public-spot exchanges that ccxt supports and that have
# (at least) BTC/USDT or BTC/USD listed. Membership is *advisory* — any ccxt
# id resolvable via `hasattr(ccxt_async, ...)` will be accepted. Listing here
# just means "we have probed this exchange and it works for the default
# symbols". The collector silently skips (exchange, symbol) pairs whose
# market is not listed, so configuring 30 exchanges with one symbol that 5
# of them don't list is fine.
SUPPORTED_EXCHANGES: tuple[str, ...] = (
    # Tier 1 — top-volume CEX with deep BTC/ETH/SOL USDT books
    "binance",
    "okx",
    "bybit",
    "kucoin",
    "gate",
    "mexc",
    "bitget",
    "htx",
    "coinbase",
    "kraken",
    # Tier 2 — solid USDT liquidity, broad pair coverage
    "bingx",
    "bitmart",
    "cryptocom",
    "lbank",
    "whitebit",
    "poloniex",
    "ascendex",
    "phemex",
    "woo",
    "bitfinex",
    "coinex",
    "bitrue",
    "hitbtc",
    "toobit",
    "hashkey",
    "upbit",
    "digifinex",
    # Tier 3 — primarily USD-quote but also list BTC/USDT
    "bitstamp",
    "gemini",
    "exmo",
)


@dataclass(frozen=True, slots=True)
class ExchangeHandle:
    """Wrapper around a ccxt client plus metadata about its capabilities."""

    exchange_id: str
    client: Any  # ccxt.async_support.Exchange or ccxt.pro.Exchange
    supports_watch_trades: bool


def _instantiate(exchange_id: str, prefer_websocket: bool, rate_limit_safety: float) -> Any:
    if exchange_id not in SUPPORTED_EXCHANGES:
        # Allow but warn via ValueError-friendly behaviour upstream.
        pass

    options = {"enableRateLimit": True}

    if prefer_websocket and _HAS_CCXT_PRO and hasattr(ccxt_pro, exchange_id):
        cls = getattr(ccxt_pro, exchange_id)
    elif hasattr(ccxt_async, exchange_id):
        cls = getattr(ccxt_async, exchange_id)
    else:
        raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt")

    client = cls(options)
    # `rateLimit` is in milliseconds.
    if hasattr(client, "rateLimit") and client.rateLimit:
        client.rateLimit = int(client.rateLimit * rate_limit_safety)
    return client


def make_handle(
    exchange_id: str,
    *,
    prefer_websocket: bool = True,
    rate_limit_safety: float = 1.2,
) -> ExchangeHandle:
    """Instantiate a ccxt client for ``exchange_id``."""

    client = _instantiate(exchange_id, prefer_websocket, rate_limit_safety)
    supports_watch = bool(
        prefer_websocket
        and _HAS_CCXT_PRO
        and ccxt_pro is not None
        and hasattr(ccxt_pro, exchange_id)
        and getattr(client, "has", {}).get("watchTrades", False)
    )
    return ExchangeHandle(
        exchange_id=exchange_id,
        client=client,
        supports_watch_trades=supports_watch,
    )


async def close_handle(handle: ExchangeHandle) -> None:
    """Close the underlying ccxt client connection (idempotent)."""

    close = getattr(handle.client, "close", None)
    if close is None:
        return
    result = close()
    # ccxt.async_support and ccxt.pro both return awaitables here.
    if hasattr(result, "__await__"):
        await result


async def has_market(handle: ExchangeHandle, symbol: str) -> bool:
    """Return ``True`` if ``handle`` exposes ``symbol``.

    Markets are loaded lazily and memoised by ccxt itself.
    """

    client = handle.client
    if not getattr(client, "markets", None):
        try:
            await client.load_markets()
        except Exception:
            return False
    markets = getattr(client, "markets", None) or {}
    return symbol in markets
