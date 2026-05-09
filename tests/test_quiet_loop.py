"""Tests for ``install_quiet_loop_handler``.

These tests construct a ``call_exception_handler`` context dict by hand (the
public interface that ``asyncio`` uses to dispatch a Future-exception event)
and assert that the installed handler swallows the noisy ``ccxt.pro``
patterns while forwarding genuine bugs to the default handler.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from lead_lag_scanner.collector import install_quiet_loop_handler


def _raise_from_file(exc: BaseException, filename: str) -> BaseException:
    """Return ``exc`` after raising it through a frame pretending to live in ``filename``.

    Compiling ``raise __exc`` with an explicit ``filename`` and executing it
    yields a traceback whose frames carry that ``co_filename``. We use this
    to fabricate "this exception came from ccxt" stacks without depending on
    ccxt at test time.
    """

    code = compile("raise __exc", filename, "exec")
    try:
        exec(code, {"__exc": exc})
    except BaseException as caught:
        return caught
    raise AssertionError("compiled raise did not propagate")  # pragma: no cover


def _drive(handler: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Invoke ``handler`` against a fake loop and report what it forwarded."""

    fake_loop = MagicMock()
    handler(fake_loop, context)
    return {
        "default_called": fake_loop.default_exception_handler.called,
        "default_args": fake_loop.default_exception_handler.call_args,
    }


def _get_handler(loop: asyncio.AbstractEventLoop) -> Any:
    handler = loop.get_exception_handler()
    assert handler is not None, "install_quiet_loop_handler did not register a handler"
    return handler


def test_handler_drops_future_never_retrieved_with_ccxt_exception() -> None:
    loop = asyncio.new_event_loop()
    try:
        install_quiet_loop_handler(loop)
        handler = _get_handler(loop)

        # Emulate a real ccxt error class without importing ccxt at test time.
        class FakeCcxtError(Exception):
            pass

        FakeCcxtError.__module__ = "ccxt.base.errors"

        result = _drive(
            handler,
            {
                "message": "Future exception was never retrieved",
                "exception": FakeCcxtError("boom"),
            },
        )
        assert result["default_called"] is False
    finally:
        loop.close()


def test_handler_forwards_unknown_message() -> None:
    loop = asyncio.new_event_loop()
    try:
        install_quiet_loop_handler(loop)
        handler = _get_handler(loop)

        result = _drive(
            handler,
            {"message": "something unrelated went wrong", "exception": ValueError("x")},
        )
        assert result["default_called"] is True
    finally:
        loop.close()


def test_handler_forwards_real_bug_even_with_suppressed_message() -> None:
    """A non-ccxt exception under a 'never retrieved' message must still surface."""

    loop = asyncio.new_event_loop()
    try:
        install_quiet_loop_handler(loop)
        handler = _get_handler(loop)

        result = _drive(
            handler,
            {
                "message": "Future exception was never retrieved",
                "exception": KeyError("oops a real bug"),
            },
        )
        assert result["default_called"] is True
    finally:
        loop.close()


def test_handler_drops_message_with_no_exception_when_pattern_matches() -> None:
    loop = asyncio.new_event_loop()
    try:
        install_quiet_loop_handler(loop)
        handler = _get_handler(loop)

        # No 'exception' key -> we cannot decide based on the type, so we let
        # asyncio's default handler decide. (This keeps unknown noise visible.)
        result = _drive(
            handler,
            {"message": "Future exception was never retrieved"},
        )
        assert result["default_called"] is True
    finally:
        loop.close()


def test_install_is_idempotent() -> None:
    loop = asyncio.new_event_loop()
    try:
        install_quiet_loop_handler(loop)
        first = _get_handler(loop)
        install_quiet_loop_handler(loop)
        second = _get_handler(loop)
        assert first is second
    finally:
        loop.close()


def test_handler_drops_builtin_exception_with_ccxt_in_traceback() -> None:
    """ccxt.pro htx.py raises AttributeError (builtins) — suppress via tb scan."""

    loop = asyncio.new_event_loop()
    try:
        install_quiet_loop_handler(loop)
        handler = _get_handler(loop)

        exc = _raise_from_file(
            AttributeError("'Client' object has no attribute 'reset'"),
            "/site-packages/ccxt/pro/htx.py",
        )

        result = _drive(
            handler,
            {
                "message": "Task exception was never retrieved",
                "exception": exc,
            },
        )
        assert result["default_called"] is False
    finally:
        loop.close()


def test_handler_forwards_builtin_exception_with_no_ccxt_frame() -> None:
    """Genuine bug in our own code (no ccxt frame) must still surface."""

    loop = asyncio.new_event_loop()
    try:
        install_quiet_loop_handler(loop)
        handler = _get_handler(loop)

        exc = _raise_from_file(
            AttributeError("'Foo' object has no attribute 'bar'"),
            "/home/ubuntu/repos/myproject/src/widget.py",
        )

        result = _drive(
            handler,
            {
                "message": "Task exception was never retrieved",
                "exception": exc,
            },
        )
        assert result["default_called"] is True
    finally:
        loop.close()
