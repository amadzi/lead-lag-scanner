"""Unit tests for collector error-classification + backoff helpers.

These cover the pure helpers (no asyncio loop, no real exchanges) that decide
whether an error is permanent, rate-limited, or transient, and how long to
sleep before the next retry. Behavioural tests for the watch/fetch loops
themselves live in the integration suite.
"""

from __future__ import annotations

import pytest

from lead_lag_scanner.collector import (
    _RATE_LIMIT_BACKOFF_SECONDS,
    _RATE_LIMIT_JITTER_FRACTION,
    _TRANSIENT_BACKOFF_SECONDS,
    _is_permanent_error,
    _is_rate_limit_error,
    _next_backoff,
)

# ---------------------------------------------------------------------------
# Permanent-error detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        # luno: ws watchTrades requires authenticated apiKey
        "luno watchTrades() requires apiKey credentials",
        # cex.io: one ws subscription per process
        "cex one symbol per instance",
        # mexc: protobuf-frame switch ccxt has not implemented
        "mexc protobuf decoder is not implemented",
        # generic NotSupported (raw exception class name surfaces in str())
        "NotSupported: this exchange does not have fetchTrades",
        # bigone: Akamai 403 page (the actual production string)
        (
            "bigone GET https://big.one/api/v3/asset_pairs/BTC-USDT/trades "
            "403 Forbidden <HTML><HEAD><TITLE>Access Denied</TITLE></HEAD>"
            "<BODY><H1>Access Denied</H1>"
            "https&#58;&#47;&#47;errors&#46;edgesuite&#46;net&#47;..."
        ),
        # Plain-text Access Denied without the HTML envelope
        "some-cex Access Denied",
        # Akamai reference URL on its own
        "errors.edgesuite.net 18.53071002 reference",
    ],
)
def test_permanent_error_matches(message: str) -> None:
    assert _is_permanent_error(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        # transient network blip
        "Connection closed by remote server, closing code 1006",
        # generic 429 — that's a rate-limit, not permanent
        '{"error":{"code":429,"message":"Too many requests"}}',
        # auth error that is not a known permanent pattern
        "Invalid signature",
        # 5xx at our origin
        "502 Bad Gateway",
    ],
)
def test_permanent_error_rejects(message: str) -> None:
    assert _is_permanent_error(RuntimeError(message)) is False


# ---------------------------------------------------------------------------
# Rate-limit detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        # fmfwio / bequant raw error body
        '{"error":{"code":429,"message":"Too many requests","description":"Too many requests"}}',
        # ccxt's RateLimitExceeded class name surfaces in str()
        "RateLimitExceeded: throttled",
        # exchange that uses 'rate limit' phrasing
        "some-cex rate limit exceeded for endpoint /api/trades",
    ],
)
def test_rate_limit_error_matches(message: str) -> None:
    assert _is_rate_limit_error(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "Connection closed by remote server, closing code 1006",
        "NotSupported: this exchange does not have fetchTrades",
        "503 Service Unavailable",
    ],
)
def test_rate_limit_error_rejects(message: str) -> None:
    assert _is_rate_limit_error(RuntimeError(message)) is False


# ---------------------------------------------------------------------------
# Backoff schedule
# ---------------------------------------------------------------------------


def test_next_backoff_transient_walks_schedule() -> None:
    """No-rate-limit branch returns the static schedule, in order."""

    idx = 0
    seen: list[float] = []
    for _ in range(len(_TRANSIENT_BACKOFF_SECONDS) + 2):
        sleep_for, idx = _next_backoff(idx)
        seen.append(sleep_for)
    # First N values match the schedule exactly; subsequent values clamp to
    # the last schedule entry.
    assert seen[: len(_TRANSIENT_BACKOFF_SECONDS)] == list(_TRANSIENT_BACKOFF_SECONDS)
    assert seen[-1] == _TRANSIENT_BACKOFF_SECONDS[-1]
    assert seen[-2] == _TRANSIENT_BACKOFF_SECONDS[-1]


def test_next_backoff_rate_limit_uses_long_schedule_with_jitter() -> None:
    """Rate-limit branch always returns >= base*0.5 of the rate-limit schedule."""

    for idx, expected_base in enumerate(_RATE_LIMIT_BACKOFF_SECONDS):
        # Run a few times to stress the jitter path. Each call must stay
        # inside ``[base * (1 - f), base * (1 + f)]``.
        for _ in range(50):
            sleep_for, _next_idx = _next_backoff(idx, rate_limited=True)
            low = expected_base * (1.0 - _RATE_LIMIT_JITTER_FRACTION)
            high = expected_base * (1.0 + _RATE_LIMIT_JITTER_FRACTION)
            assert low <= sleep_for <= high, (
                f"jittered sleep {sleep_for} outside [{low}, {high}] for base {expected_base}"
            )


def test_next_backoff_rate_limit_floor_is_much_higher_than_transient() -> None:
    """The first rate-limit sleep is always >= 4x the first transient sleep.

    This is the actual user-facing guarantee: when 28 symbols on the same
    exchange all hit 429, the next round of retries is spread out over
    tens of seconds (not 1s), giving the exchange room to recover.
    """

    # Worst case: maximum negative jitter on rate-limit, raw transient value.
    rate_limit_min = _RATE_LIMIT_BACKOFF_SECONDS[0] * (1.0 - _RATE_LIMIT_JITTER_FRACTION)
    transient_max = _TRANSIENT_BACKOFF_SECONDS[0]
    assert rate_limit_min >= transient_max * 4
