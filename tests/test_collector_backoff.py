"""Unit tests for collector error-classification + backoff helpers.

These cover the pure helpers (no asyncio loop, no real exchanges) that decide
whether an error is permanent, rate-limited, or transient, and how long to
sleep before the next retry. Behavioural tests for the watch/fetch loops
themselves live in the integration suite.
"""

from __future__ import annotations

import pytest

from lead_lag_scanner.collector import (
    _MAX_CONSECUTIVE_FAILURES,
    _RATE_LIMIT_BACKOFF_SECONDS,
    _RATE_LIMIT_JITTER_FRACTION,
    _TRANSIENT_BACKOFF_SECONDS,
    _is_permanent_error,
    _is_rate_limit_error,
    _next_backoff,
    _normalise_error_text,
)

# ---------------------------------------------------------------------------
# Permanent-error detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        # luno: ws watchTrades requires authenticated apiKey (no quotes)
        "luno watchTrades() requires apiKey credentials",
        # luno: actual production ccxt string with double-quoted apiKey —
        # caught by the quote-stripped match path.
        'luno requires "apiKey" credential',
        # ccxt variant with backticks (some older releases)
        "luno requires `apiKey` credential",
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
        # weex: invalid contract / INVALID_ARGUMENT for an unsupported symbol
        'weex {"result":false,"id":278,"msg":"INVALID_ARGUMENT: invalid contract"}',
        "weex INVALID_ARGUMENT: invalid contract for SOL/USDT",
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
        # bitget — application-level 30006 rate-limit code
        'bitget {"event":"error","code":30006,"msg":"request too many"}',
        # bitmart — 90006 (total topic quantity exceeds limit)
        (
            'bitmart {"errorMessage":"Subscribed total topic quantity exceeds limit",'
            '"errorCode":"90006","event":"subscribe"}'
        ),
        # bitmart — 90007 (frequency exceeds limit)
        (
            'bitmart {"errorMessage":"Subscribed message frequency exceeds limit, '
            'please try later","errorCode":"90007","event":"..."}'
        ),
        # bigone — application-level 10429
        'bigone {"code":10429,"message":"Too many requests"}',
        # coinsph — application-level -1003
        'coinsph {"code":-1003,"msg":"Too many requests; current request has limited."}',
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


# ---------------------------------------------------------------------------
# Quote-stripping match path (the luno fix)
# ---------------------------------------------------------------------------


def test_normalise_error_text_strips_quotes() -> None:
    """``_normalise_error_text`` returns lowered + lowered-without-quotes."""

    lowered, no_quotes = _normalise_error_text(RuntimeError('luno requires "apiKey" credential'))
    assert lowered == 'luno requires "apikey" credential'
    assert no_quotes == "luno requires apikey credential"


def test_normalise_error_text_handles_backticks_and_singles() -> None:
    lowered, no_quotes = _normalise_error_text(
        RuntimeError("ws subscribe to one `symbol` per inst")
    )
    assert "`" in lowered
    assert "`" not in no_quotes
    # the canonical pattern matches the stripped form even though the
    # raw form has backticks in the middle of the phrase.
    assert "subscribe to one symbol" in no_quotes


# ---------------------------------------------------------------------------
# Consecutive-failure threshold
# ---------------------------------------------------------------------------


def test_max_consecutive_failures_is_small_enough_to_avoid_log_spam() -> None:
    """Threshold should mute a stuck (exchange, symbol) pair within a small
    constant number of WARN lines.

    The user-facing guarantee: a permanently-broken pair like upbit's
    keepalive cycle produces at most ``_MAX_CONSECUTIVE_FAILURES`` WARN
    lines (one per retry) plus exactly one INFO line, regardless of how
    long the run lasts. With 50 symbols on a stuck exchange that's
    ``50 * (K + 1)`` lines total — flat in run duration.
    """

    assert 1 < _MAX_CONSECUTIVE_FAILURES <= 10


def test_max_consecutive_failures_is_large_enough_to_ride_out_brief_blips() -> None:
    """K + the transient schedule must give at least ~50s of patience.

    A symbol whose exchange has a brief blip will see the loop retry
    through the schedule and recover, resetting the counter to 0. The
    transient schedule sums to ``1 + 5 + 15 + 30 = 51s`` for K=5
    (the first four sleep intervals), enough to ride out a typical
    DNS / TLS / TCP blip without silently muting exchanges.
    """

    # Sum of the first K-1 entries of the transient schedule (the time
    # between error #1 and the K-th error, inclusive).
    transient_total = 0.0
    for i in range(_MAX_CONSECUTIVE_FAILURES - 1):
        idx = min(i, len(_TRANSIENT_BACKOFF_SECONDS) - 1)
        transient_total += _TRANSIENT_BACKOFF_SECONDS[idx]
    assert transient_total >= 50.0
